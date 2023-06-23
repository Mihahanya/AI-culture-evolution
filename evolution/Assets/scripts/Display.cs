using Accord.Math;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions.Must;
using UnityEngine.Experimental.PlayerLoop;
using UnityEngine.Rendering;
using UnityEngine.UI;
using static UnityEditor.PlayerSettings;
using static UnityEngine.ParticleSystem;
using static UnityEngine.UI.Image;


public class Display : MonoBehaviour
{
    public GameObject panel;
    public GameObject neuronOrient;
    public GameObject weightsOrient;
    public Text dataOut;
    public Text skillData;
    public Text agentsCounter;
    public Text fpsText;
    public Text epochDurationText;
    public GameObject lineBase;

    public float neuronsRadius = 0.1f;

    public bool isFollowObject = false;    
    public GameObject followingObject;

    public GameObject circle;
    GameObject[] neuronsViss;

    Vector2 visualizationSizes;

    float timer = 0.1f;
    int bacteriaCount = 0, prevBacteriaCount = 0;
    int deaths, borns;

    float frameTimer = 1f;
    int framesCount = 0;
    
    void Start()
    {
        Application.runInBackground = true;

        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = Config.fps;
        epochDurationText.text = Config.epochDuration.ToString();

        camera cam = GetComponent<camera>();

        panel.SetActive(false);

        var orient = neuronOrient.GetComponent<RectTransform>();
        visualizationSizes = orient.sizeDelta;
    }

    void Update()
    {
        var bacterias = GameObject.FindGameObjectsWithTag("bacteria");
        bacteriaCount = bacterias.Length;

        if (bacteriaCount < prevBacteriaCount)
            deaths += prevBacteriaCount - bacteriaCount;
        else if (bacteriaCount > prevBacteriaCount)
            borns += bacteriaCount - prevBacteriaCount;

        prevBacteriaCount = bacteriaCount;

        
        timer -= Time.deltaTime;
        if (timer < 0)
        {
            float midAge = 0, midGeneration = 0;

            for (int i = 0; i < bacterias.Length; i++)
            {
                var bacData = bacterias[i].GetComponent<Agent>();

                midAge += bacData.age;
                midGeneration += bacData.generation;
            }

            midAge /= bacterias.Length;
            midGeneration /= bacterias.Length;

            agentsCounter.text = "";
            agentsCounter.text += "Bacteria count: " + bacteriaCount + "\n";
            agentsCounter.text += "Born : " + borns + "\n";
            agentsCounter.text += "Death: " + deaths + "\n";
            agentsCounter.text += "Middle Age: " + midAge + "\n";
            agentsCounter.text += "Middle Generation: " + midGeneration + "\n";

            borns = 0;
            deaths = 0;

            timer = 2f;
        }

    }

    void LateUpdate()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Vector2 origin = new Vector2(Camera.main.ScreenToWorldPoint(Input.mousePosition).x,
                                         Camera.main.ScreenToWorldPoint(Input.mousePosition).y) + Vector2.one * 0.01f;
            RaycastHit2D hit = Physics2D.Raycast(origin, Vector2.zero, 0f);
            if (hit)
            {
                if (hit.transform.tag == "bacteria")
                {
                    if (isFollowObject) Close();

                    followingObject = hit.transform.gameObject;
                    isFollowObject = true;

                    panel.SetActive(true);

                    string skillsTextData = "Skills data:\n";

                    var objData = followingObject.GetComponent<Agent>();

                    foreach (var skill in objData.genome.skills)
                        skillsTextData += skill.Key + ": " + (skill.Value.First * skill.Value.Second) + "\n";


                    var nn = objData.genome.nn;
                    UpdateWeights(nn);
                    neuronsViss = DrawReturnNeurons(nn);

                    skillData.text = skillsTextData;
                }
                else Close();
            }
            else Close();
        }


        if (isFollowObject && followingObject == null) isFollowObject = false;


        if (isFollowObject)
        {
            var objData = followingObject.GetComponent<Agent>();

            var layers = objData.genome.nn.layers;

            int t = 0;
            for (t = 0; t < objData.inputs.Length; t++)
            {
                var input = (float)(objData.inputs[t] + 1f) / 2f;
                neuronsViss[t].GetComponent<SpriteRenderer>().color = new Color(1f - input, input, 0.5f) / 1.05f;
            }

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].neurons.Length; j++)
                {
                    var neuron = (float)layers[i].neurons[j];
                    var neuronInterpolated = (neuron + 1f) / 2f;

                    neuronsViss[t].GetComponent<SpriteRenderer>().color = new Color(1f - neuronInterpolated, neuronInterpolated, 0.5f) / 1.05f;

                    t++;
                }
            }

            if (objData.epoch.IsDone()) UpdateWeights(objData.genome.nn);

            dataOut.text = "";
            dataOut.text += "Energy: " + objData.energy + "\n";
            dataOut.text += "Age: " + objData.age + "\n";
            dataOut.text += "Generation: " + objData.generation + "\n";
        }


        if (Input.GetKeyDown(KeyCode.F))
        {
            Config.SetEpochDuration(Config.epochDuration / 1.2f);
            Application.targetFrameRate = Config.fps;
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            Config.SetEpochDuration(Config.epochDuration * 1.2f);
            Application.targetFrameRate = Config.fps;
        }


        frameTimer -= Time.deltaTime;
        if (frameTimer < 0)
        {
            fpsText.text = "Steps per second: " + framesCount;
            epochDurationText.text = "Epoch duration:" + (Config.epochDuration).ToString("N4") + 
                                     ", Actual: " + ((float)Config.stepsPerEpoch / framesCount).ToString("N4");

            framesCount = 0;
            frameTimer = 1f;
        }
        framesCount++;
    }

    GameObject[] DrawReturnNeurons(NN nn)
    {
        int[] sizes = nn.sizes;

        GameObject[] neuronsViss = new GameObject[sizes.Sum()];

        int t = 0;
        for (int i = 0; i < sizes.Length; i++)
        {
            for (int j = 0; j < sizes[i] + (i < sizes.Length-1 ? 1 : 0); j++)
            {
                GameObject neuronVisual = (GameObject)UnityEngine.Object.Instantiate(circle);
                neuronVisual.transform.parent = neuronOrient.transform;
                neuronVisual.transform.localScale = UnityEngine.Vector3.one * neuronsRadius;
                neuronVisual.GetComponent<SpriteRenderer>().color = new Color(0f, 1f, 0.5f);

                Vector2 pos = new Vector2(i * visualizationSizes.x / (sizes.Length-1), j * visualizationSizes.y / (sizes[i]-1+1)); // +1 because bias

                neuronVisual.AddComponent<RectTransform>();
                RectTransform p = neuronVisual.GetComponent<RectTransform>();
                p.anchoredPosition = pos - visualizationSizes / 2f;
                p.anchoredPosition3D = new UnityEngine.Vector3(p.anchoredPosition.x, -p.anchoredPosition.y, 0f);

                if (j != sizes[i])
                {
                    neuronsViss[t] = neuronVisual;
                    t++;
                }
            }
            
        }

        return neuronsViss;
    }

    void UpdateWeights(NN nn)
    {
        foreach (Transform w in weightsOrient.transform)
            Destroy(w.gameObject);


        int[] sizes = nn.sizes;

        // Weight max deviation

        double amplitude = 0;
        for (int i = 0; i < nn.layers.Length; i++)
        {
            double currentAmplitude = Matrix.Max(nn.layers[i].weights);
            currentAmplitude = Math.Max(Matrix.Min(nn.layers[i].weights), currentAmplitude);

            amplitude = Math.Max(amplitude, currentAmplitude);
        }

        // Weights lines

        for (int i = 0; i < nn.layers.Length; i++) // Layers
        {
            var weights = nn.layers[i].weights;
            for (int wi = 0; wi < sizes[i + 1]; wi++) // Ends
            {
                for (int wj = 0; wj < sizes[i] + 1; wj++) // Starts
                {
                    UnityEngine.Vector3 start = new UnityEngine.Vector3(i * visualizationSizes.x / (sizes.Length - 1),
                                                wj * visualizationSizes.y / (sizes[i] - 1 + 1), 0f);

                    UnityEngine.Vector3 end = new UnityEngine.Vector3((i + 1) * visualizationSizes.x / (sizes.Length - 1),
                                              wi * visualizationSizes.y / (sizes[i + 1] - 1 + 1), 0f);

                    double w;
                    if (wj == sizes[i]) w = nn.layers[i].biases[wi];
                    else w = weights[wi, wj];

                    float aspect = (float)((float)w / amplitude);
                    float thickness = Mathf.Max(Mathf.Abs(aspect) * neuronsRadius, 2f);

                    float aspectInterpolated = (aspect + 1f) / 2f;

                    DrawLineRectTransform(start, end, thickness, new Color(1f - aspectInterpolated, aspectInterpolated, 0.5f) / 2f);
                }
            }
        }
    }


    void DrawLineRectTransform(Vector2 start, Vector2 end, float width, Color color)
    {
        GameObject myLine = (GameObject)UnityEngine.Object.Instantiate(lineBase);
        //myLine.transform.parent = weightsOrient.transform;

        RectTransform nnp = weightsOrient.GetComponent<RectTransform>();
        RectTransform p = myLine.GetComponent<RectTransform>();
        p.SetParent(nnp);
        p.anchoredPosition = Vector2.zero;
        p.sizeDelta = nnp.sizeDelta;
        p.localScale = UnityEngine.Vector3.one;
        p.anchoredPosition3D = new UnityEngine.Vector3(p.anchoredPosition.x, p.anchoredPosition.y, 1f);

        LineRenderer lr = myLine.GetComponent<LineRenderer>();
        lr.SetColors(color, color);
        lr.SetWidth(width, width);

        lr.SetPosition(0, (start - visualizationSizes / 2f) * new Vector2(1f, -1f));
        lr.SetPosition(1, (end - visualizationSizes / 2f) * new Vector2(1f, -1f));
    }


    void Close()
    {
        foreach (Transform neur in neuronOrient.transform)
            Destroy(neur.gameObject);
    
        foreach (Transform w in weightsOrient.transform)
            Destroy(w.gameObject);
    
        isFollowObject = false;
        panel.SetActive(false);
    }

}
