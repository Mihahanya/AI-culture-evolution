using System;
using UnityEngine;
using System.Linq;
using Accord;

public class Agent : MonoBehaviour
{
    [NonSerialized]
    public bool exhaustion = false;
    [NonSerialized]
    public bool encourage = true;

    public GameObject bacteriaPrefab;
    Rigidbody2D rb;

    public bool watchingByCamera = false;

    [NonSerialized]
    public float energy = 140;
    public float age = 0;
    public int generation = 0;

    public Genome genome;
    public Epoch epoch;

    private const int eyesCount = 6;
    private const int memoryNeuronsN = 5;
                                // eyes by distance and colors, memory, (~age/energy, ~energy/age) 
    private const int inputCount = eyesCount * 4 + memoryNeuronsN + 2; 
    private const int outputCount = 3 + memoryNeuronsN + 1; // move, divide

    float speed, angSpeed, foodProd, memoryFactor, needEnergyToDivide;
    float speedK, angSpeedK;

    [NonSerialized]
    public double[] inputs;
    [NonSerialized]
    public double[] outputs;
    [NonSerialized]
    public double[] memoryNeuronsState;
    [NonSerialized]
    public float reward = 0;
    [NonSerialized]
    public float punishing = 0;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();

        inputs = new double[inputCount];
        outputs = new double[outputCount];
        memoryNeuronsState = new double[memoryNeuronsN];

        if (generation == 0) genome = new Genome(inputCount, outputCount);

        InitAgent();
    }

    public void InitAgent()
    {
        GetComponent<SpriteRenderer>().color = new Color(genome.r, genome.g, genome.b);

        speed = genome.GetActualSkill("speed");
        angSpeed = genome.GetActualSkill("angularSpeed");
        memoryFactor = genome.GetActualSkill("memoryFactor");
        needEnergyToDivide = genome.GetActualSkill("needEnergyDivide");
        //var size = genome.GetActualSkill("size");
        //transform.localScale = Vector3.one * size;

        speedK = genome.skills["speed"].First;
        angSpeedK = genome.skills["angularSpeed"].First;

        epoch = new Epoch(Config.stepsPerEpoch);
    }

    void LateUpdate()
    {
        bool pause = Camera.main.GetComponent<camera>().pause;
        if (pause) return;

        age++;

        if (exhaustion) energy -= 1f;

        // Setting new inputs

        var newInputs = new double[] { };

        // Age adn hunger data
        newInputs = newInputs.Concat(new double[] { age / energy, energy / age }).ToArray();

        // Memory
        newInputs = newInputs.Concat(memoryNeuronsState).ToArray();

        double[] visionData = new double[eyesCount*4];
        float maxDist = 15f;

        for (int i = 0; i < eyesCount; i++)
        {
            Vector2 direction = transform.up.xy().RotateVector(i * 360f / eyesCount);
            Vector2 from = new Vector2(transform.position.x, transform.position.y) + direction * 2f * transform.localScale.x;
            RaycastHit2D hit = Physics2D.Raycast(from, direction, maxDist);
            if (hit)
            {
                visionData[i] = 1f - hit.distance / maxDist;

                Color hitColor = hit.transform.gameObject.GetComponent<SpriteRenderer>().color;

                visionData[i*4 + 1] = hitColor.r;
                visionData[i*4 + 2] = hitColor.g;
                visionData[i*4 + 3] = hitColor.b;

                Debug.DrawRay(from, hit.distance * direction, hitColor);
            }
            else
            {
                visionData[i] = 0;
                visionData[i * 4 + 1] = -1;
                visionData[i * 4 + 2] = -1;
                visionData[i * 4 + 3] = -1;
            }
        }

        newInputs = newInputs.Concat(visionData).ToArray();

        Debug.Assert(newInputs.Length == inputCount);

        // Update inputs

        for (int i = 0; i < inputCount; i++)
        {
            inputs[i] = inputs[i] * (1 - memoryFactor) + newInputs[i] * memoryFactor;
        }

        // Decode outputs

        outputs = genome.nn.FeedForward(inputs);

        Vector2 movOut = new Vector2((float)outputs[0], (float)outputs[1]).normalized;
        float rotOut = (float)outputs[2];
        bool isDivideOut = outputs[3] > 0.5;
        memoryNeuronsState = outputs.Skip(4).Take(memoryNeuronsN).ToArray();

        // Physical result

        float fps = Config.fps;
        if (Config.fps > 1f / Time.deltaTime) fps = 1f / Time.deltaTime;

        rb.MovePosition(transform.position.xy() + 
                        (transform.up.xy() * movOut.y + transform.right.xy() * movOut.x) * speed);
        
        rb.MoveRotation(rb.rotation + rotOut * angSpeed);
        
        if (energy >= needEnergyToDivide)
        //if (isDivideOut && energy >= needEnergyToDivide)
            DivideYourself();

        // Movement coasts

        if (exhaustion)
        {
            var consumption = 0f;
            consumption += movOut.magnitude * Mathf.Abs(speedK);
            consumption += rotOut * Mathf.Abs(angSpeedK);
        
            //consumption *= Mathf.Pow(genome.skills["size"].First, 2f) * 0.35f;
            consumption *= 0.3f;

            energy -= consumption;
        }

        // Death

        if (energy <= 0)
        {
            Camera.main.GetComponent<Display>().deathCount++;
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
            //
            //reward += (1f / (minDist + 1) - 1f / (5 + 1)) * 0.05f;

            //reward += findedRays_n / eyesCount * 0.001f;

            epoch.AddEpoch(inputs, genome.nn.GetState(), reward, punishing);

            if (epoch.IsDone())
            {
                epoch.Apply(ref genome.nn);
                
                if (watchingByCamera) Camera.main.GetComponent<Display>().UpdateWeights(genome.nn);
            }
        }
        reward = 0f;
        punishing = 0f;
    }

    void OnTriggerEnter2D(Collider2D col)
    {
        if (col.gameObject.tag == "food")
        {
            //var fd = genome.GetActualSkill("foodAspect");
            reward += 1f;
            energy += 2.5f*Config.stepsPerEpoch;
            Destroy(col.gameObject);
        }
    }

    void DivideYourself()
    {
        return;

        reward += 3;

        GameObject b = (GameObject)UnityEngine.Object.Instantiate(bacteriaPrefab);
        b.transform.position = transform.position;

        var agentMind = b.GetComponent<Agent>();

        agentMind.generation = generation + 1;
        agentMind.age = 0;

        agentMind.genome = new Genome(genome);
        if (UnityEngine.Random.value < 0.5) 
            agentMind.genome.Mutate(0.5f, 0.2f, 0.2f, 0.05f);
        else
            agentMind.genome.Mutate(0.2f, 0.7f, 0.2f, 0.05f);

        agentMind.InitAgent();

        agentMind.energy = energy / 2f;
        energy /= 2f;

        Camera.main.GetComponent<Display>().bornCount++;
    }

    void OnCollisionEnter2D(Collision2D col)
    {
        if (col.gameObject.tag == "wall")
        {
            punishing += 0.5f;
            transform.position = new Vector3(-transform.position.x, -transform.position.y, transform.position.z) + transform.position.normalized;
        }
    }

}
