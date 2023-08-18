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
    public float energy = 14 * Config.stepsPerEpoch;
    public float age = 0;
    public int generation = 0;

    public Genome genome;
    public Epoch epoch_of_act;
    public Epoch epoch_of_rew;

    private const int eyesCount = 10;
    private const int memoryNeuronsN = 5;
    
    // move, memory, is divide, is eat, is attack
    private const int outputCount = 3 + memoryNeuronsN + 1 + 1 + 1; 

    // eyes by distance and colors, (age/energy, energy/age), pain, body sensation
    private const int inputCount = eyesCount * 4 + 2 + 1 + outputCount;
    
    float speed, angSpeed, memoryFactor, needEnergyToDivide;
    float speedK, angSpeedK;
    float nonLiveFoodAssimil, liveFoodAssimil;
    
    public double[] memoryNeuronsState;
    bool isDivideOut, isEatOut, isAttackOut;
    float eatOut, attackOut;

    [NonSerialized]
    public double[] inputs;
    [NonSerialized]
    public double[] outputs;

    public float reward = 0;
    public float punishing = 0;
    public float pain;

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

        nonLiveFoodAssimil = genome.GetActualSkill("foodAspect");
        liveFoodAssimil = 1f - nonLiveFoodAssimil; // how predatory is bacteria

        speedK = genome.skills["speed"].First;
        angSpeedK = genome.skills["angularSpeed"].First;

        epoch_of_act = new Epoch(Config.stepsPerEpoch);
        epoch_of_rew = new Epoch(Config.stepsPerEpoch);
    }

    void LateUpdate()
    {
        bool pause = Camera.main.GetComponent<camera>().pause;
        if (pause) return;

        age++;

        if (exhaustion) energy -= 1f;
        punishing += pain*2f;

        // Setting new inputs

        // Pain, age adn hunger data
        var newInputs = new double[] { pain, age / energy, energy / age };

        // World vision
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

        // Body sensation and memory
        newInputs = newInputs.Concat(outputs).ToArray();

        Debug.Assert(newInputs.Length == inputCount);

        // Update inputs

        for (int i = 0; i < inputCount; i++)
        {
            inputs[i] = inputs[i] * (1 - memoryFactor) + newInputs[i] * memoryFactor;
        }

        // Decode outputs

        outputs = genome.actNN.FeedForward(inputs);

        Vector2 movOut = new Vector2((float)outputs[0], (float)outputs[1]).normalized;
        float rotOut = (float)outputs[2];
        isDivideOut = outputs[3] > 0d;

        eatOut = (float)outputs[4];
        attackOut = (float)outputs[5];
        isEatOut = eatOut > 0f;
        isAttackOut = attackOut > 0f;
        
        memoryNeuronsState = outputs.Skip(6).Take(memoryNeuronsN).ToArray();

        // Physical result

        float fps = Config.fps;
        if (Config.fps > 1f / Time.deltaTime) fps = 1f / Time.deltaTime;

        rb.MovePosition(transform.position.xy() + 
                        (transform.up.xy() * movOut.y + transform.right.xy() * movOut.x) * speed);
        
        rb.MoveRotation(rb.rotation + rotOut * angSpeed);
        
        //if (energy >= needEnergyToDivide)
        //if (isDivideOut && energy >= needEnergyToDivide)
        //    DivideYourself();

        // Movement coasts

        if (exhaustion)
        {
            var consumption = 0f;
            consumption += movOut.magnitude * Mathf.Abs(speedK);
            consumption += Mathf.Abs(rotOut) * Mathf.Abs(angSpeedK);
            if (isEatOut) consumption += eatOut * nonLiveFoodAssimil * 0.1f;
            if (isAttackOut) consumption += attackOut * liveFoodAssimil * 0.7f;
        
            //consumption *= Mathf.Pow(genome.skills["size"].First, 2f) * 0.35f;
            consumption *= 0.4f;

            energy -= consumption;
            punishing += consumption * 0.05f;
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

            var act_nn_state = genome.actNN.GetState().SelectMany(subArray => subArray).ToArray();
            act_nn_state = inputs.Concat(act_nn_state).ToArray();

            var rew_pun = genome.rewNN.FeedForward(act_nn_state);

            epoch_of_act.AddEpoch(inputs, genome.actNN.GetState(), rew_pun[0], rew_pun[1]);
            
            epoch_of_rew.AddEpoch(act_nn_state, genome.rewNN.GetState(), reward, punishing);

            if (epoch_of_act.IsDone())
            {
                epoch_of_act.Apply(ref genome.actNN);
                epoch_of_rew.Apply(ref genome.rewNN);

                if (watchingByCamera)
                {
                    Camera.main.GetComponent<Display>().drawActNN.Visualize(genome.actNN);
                    Camera.main.GetComponent<Display>().drawRewNN.Visualize(genome.rewNN);
                }
            }
        }
        reward = 0f;
        punishing = 0f;
        pain = 0f;
    }

    void OnTriggerEnter2D(Collider2D col)
    {
        if (col.gameObject.tag == "food" && isEatOut)
        {
            float k = nonLiveFoodAssimil * eatOut;
            reward += 1f * k;
            energy += 2.5f * Config.stepsPerEpoch * k;
            Destroy(col.gameObject);
        }
        
        if (col.gameObject.tag == "bacteria" && isAttackOut)
        {
            reward += 1f * liveFoodAssimil * attackOut;

            var pray = col.gameObject.GetComponent<Agent>();
            //float getting = 2.5f * Config.stepsPerEpoch * k;
            float injury = pray.energy / 2f * attackOut;
            energy += injury * liveFoodAssimil;
            pray.energy -= injury;
            pray.pain += attackOut;
        }
    }

    void DivideYourself()
    {
        //return;

        reward += 7;

        GameObject b = (GameObject)UnityEngine.Object.Instantiate(bacteriaPrefab);
        b.transform.position = transform.position;

        var agentMind = b.GetComponent<Agent>();

        agentMind.generation = generation + 1;
        agentMind.age = 0;

        agentMind.genome = new Genome(genome);
        if (UnityEngine.Random.value < 0.5) 
            agentMind.genome.Mutate(0.5f, 0.2f, 0.2f, 0.1f);
        else
            agentMind.genome.Mutate(0.2f, 0.7f, 0.2f, 0.1f);

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
