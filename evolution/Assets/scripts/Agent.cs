using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Agent : MonoBehaviour
{
    [NonSerialized]
    public bool exhaustion = false;
    [NonSerialized]
    public bool encourage = true;

    public GameObject bacteriaPrefab;
    Rigidbody2D rb;

    [NonSerialized]
    public float energy = 300;
    public float age = 0;
    public int generation = 0;

    public Genome genome;
    public Epoch epoch;

    private const int eyesCount = 6;
    private const int memoryNeurons = 0;
    private const int inputCount = eyesCount + 3 + memoryNeurons + 0; // eyes by distance and colors, ~age/energy, ~energy/age 
    private const int outputCount = 3 + memoryNeurons + 1; // move, divide

    float speed, angSpeed, foodProd, memoryFactor;

    [NonSerialized]
    public float[] inputs;
    [NonSerialized]
    public float[] outputs;
    [NonSerialized]
    public float reward = 0;


    void Start()
    {
        rb = GetComponent<Rigidbody2D>();

        inputs = new float[inputCount];
        outputs = new float[outputCount];

        epoch = new Epoch(Config.stepsPerEpoch);

        if (generation == 0) genome = new Genome(inputCount, outputCount);

        InitAgent();
    }

    public void InitAgent()
    {
        GetComponent<SpriteRenderer>().color = new Color(genome.r, genome.g, genome.b);

        speed = genome.GetActualSkill("speed");
        angSpeed = genome.GetActualSkill("angularSpeed");
        memoryFactor = genome.GetActualSkill("memoryFactor");
        //var size = genome.GetActualSkill("size");
        //transform.localScale = Vector3.one * size;
    }

    void LateUpdate()
    {
        bool pause = Camera.main.GetComponent<camera>().pause;
        if (pause) return;

        age++;

        if (exhaustion) energy -= 1f;


        float[] newInputs = new float[inputCount];

        float maxDist = 15f;

        float midColR = 0f;
        float midColG = 0f;
        float midColB = 0f;
        int findedRays_n = 0;

        for (int i = 0; i < eyesCount; i++)
        {
            Vector2 direction = RotateVector(transform.up, i * 360f / eyesCount);
            Vector2 from = new Vector2(transform.position.x, transform.position.y) + direction * 2f * transform.localScale.x;
            RaycastHit2D hit = Physics2D.Raycast(from, direction, maxDist);
            if (hit)
            {
                newInputs[i] = 1f - hit.distance / maxDist;

                Color hitColor = hit.transform.gameObject.GetComponent<SpriteRenderer>().color;

                //newInputs[i*4 + 1] = hitColor.r;
                //newInputs[i*4 + 2] = hitColor.g;
                //newInputs[i*4 + 3] = hitColor.b;

                findedRays_n++;
                midColR += hitColor.r;
                midColG += hitColor.g;
                midColB += hitColor.b;

                Debug.DrawRay(from, hit.distance * direction, hitColor);
            }
            else
            {
                newInputs[i] = 0;
                //newInputs[i * 4 + 1] = -1;
                //newInputs[i * 4 + 2] = -1;
                //newInputs[i * 4 + 3] = -1;
            }
        }

        if (findedRays_n > 0)
        {
            newInputs[eyesCount + 0] = midColR / findedRays_n;
            newInputs[eyesCount + 1] = midColG / findedRays_n;
            newInputs[eyesCount + 2] = midColB / findedRays_n;
        }
        else
        {
            newInputs[eyesCount + 0] = -1;
            newInputs[eyesCount + 1] = -1;
            newInputs[eyesCount + 2] = -1;
        }

        for (int i = 0; i < memoryNeurons; i++)
        {
            newInputs[eyesCount + 3 + i] = outputs[i + 4]; // because movement, rotation and division
        }

        //newInputs[eyesCount + 3 + memoryNeurons] = age / energy;
        //newInputs[eyesCount + 3 + memoryNeurons + 1] = energy / age;

        for (int i = 0; i < inputCount; i++)
        {
            inputs[i] = inputs[i] * (1 - memoryFactor) + newInputs[i] * memoryFactor;
        }

        outputs = genome.nn.FeedForward(inputs);

        // Physical result
        
        float fps = Config.fps;
        if (Config.fps > 1f / Time.deltaTime) fps = 1f / Time.deltaTime;

        rb.velocity = RotateVector(new Vector2(outputs[0], outputs[1]) * speed, transform.localEulerAngles.z) * fps / Config.stepsPerEpoch;
        rb.angularVelocity = outputs[2] * angSpeed * fps / Config.stepsPerEpoch;

        if (outputs[3] > 0.5f && energy >= genome.GetActualSkill("needEnergyDivide"))
            DivideYourself();

        // Movement coasts

        if (exhaustion)
        {
            var consumption = 0f;
            consumption += new Vector2(outputs[0], outputs[1]).magnitude * Mathf.Abs(genome.skills["speed"].First);
            consumption += Mathf.Abs(outputs[2]) * Mathf.Abs(genome.skills["angularSpeed"].First);
        
            //consumption *= Mathf.Pow(genome.skills["size"].First, 2f) * 0.35f;
            consumption *= 0.2f;

            energy -= consumption;
        }

        // Death

        if (energy <= 0)
        {
            Destroy(gameObject);
        }

        // Rewarding

        if (encourage)
        {
            //var foods = GameObject.FindGameObjectsWithTag("food");
            //float minDist = 5;
            //foreach (var f in foods)
            //{
            //    float d = Vector2.Distance(f.transform.position, transform.position);
            //    if (d < minDist)
            //        minDist = d;
            //}

            //reward += (1f / (minDist + 1) - 1f / (5 + 1)) * 0.02f;

            //reward += findedRays_n / eyesCount * 0.001f;

            epoch.AddEpoch(inputs, outputs, reward);

            if (epoch.IsDone()) epoch.Apply(ref genome.nn);
        }
        reward = 0;
    }

    void OnTriggerEnter2D(Collider2D col)
    {
        if (col.gameObject.tag == "food")
        {
            //var fd = genome.GetActualSkill("foodAspect");
            reward += 1;
            energy += 2.5f*Config.stepsPerEpoch;
            Destroy(col.gameObject);
        }
    }

    void DivideYourself()
    {
        return;

        reward += 7;

        GameObject b = (GameObject)UnityEngine.Object.Instantiate(bacteriaPrefab);
        b.transform.position = transform.position;

        var agentMind = b.GetComponent<Agent>();

        agentMind.generation = generation + 1;
        agentMind.age = 0;

        agentMind.genome = new Genome(genome);
        agentMind.genome.Mutate(0.5f, 0.3f, 0.2f, 0.05f);

        agentMind.InitAgent();

        agentMind.energy = energy / 2f;
        energy /= 2f;
    }

    void OnCollisionEnter2D(Collision2D col)
    {
        if (col.gameObject.tag == "wall")
        {
            transform.position = new Vector3(-transform.position.x, -transform.position.y, transform.position.z) + transform.position.normalized;
        }
    }


    public Vector2 RotateVector(Vector2 v, float angle)
    {
        float radian = angle * Mathf.Deg2Rad;
        float _x = v.x * Mathf.Cos(radian) - v.y * Mathf.Sin(radian);
        float _y = v.x * Mathf.Sin(radian) + v.y * Mathf.Cos(radian);
        return new Vector2(_x, _y);
    }
}
