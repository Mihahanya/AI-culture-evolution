using Accord.Math;
using System;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;


public class Display : MonoBehaviour
{
    public GameObject panel;
    public Text dataOut;
    public Text skillData;
    public Text agentsCounter;
    public Text fpsText;
    public Text epochDurationText;

    public bool isFollowObject = false;    
    public GameObject followingObject;

    public DrawNN drawRewNN;
    public DrawNN drawActNN;

    float timer = 10;
    public int bacteriaCount = 0;
    public int deathCount = 0;
    public int bornCount = 0;

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
    }

    void Update()
    {
        var bacterias = GameObject.FindGameObjectsWithTag("bacteria");
        bacteriaCount = bacterias.Length;
        
        //timer -= Time.deltaTime;
        timer--;
        if (timer <= 0)
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
            agentsCounter.text += "Born : " + bornCount + "\n";
            agentsCounter.text += "Death: " + deathCount + "\n";
            agentsCounter.text += "Middle Age: " + midAge + "\n";
            agentsCounter.text += "Middle Generation: " + midGeneration + "\n";

            bornCount = 0;
            deathCount = 0;

            timer = Config.stepsPerEpoch;
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

                    objData.watchingByCamera = true;

                    foreach (var skill in objData.genome.skills)
                        skillsTextData += skill.Key + ": " + (skill.Value.First * skill.Value.Second) + "\n";

                    skillData.text = skillsTextData;

                    drawActNN.Visualize(objData.genome.actNN);
                    drawRewNN.Visualize(objData.genome.rewNN);
                }
                else Close();
            }
            else Close();
        }


        if (isFollowObject && followingObject == null) isFollowObject = false;


        if (isFollowObject)
        {
            var objData = followingObject.GetComponent<Agent>();

            dataOut.text = "";
            dataOut.text += "Energy: " + objData.energy + "\n";
            dataOut.text += "Age: " + objData.age + "\n";
            dataOut.text += "Generation: " + objData.generation + "\n";

            dataOut.text += "Shape: ";
            foreach (var s in objData.genome.actNN.sizes) dataOut.text += s.ToString() + " ";
            dataOut.text += "\n";
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

    void Close()
    {
        if (followingObject != null) followingObject.GetComponent<Agent>().watchingByCamera = false;

        drawActNN.Close();
        drawRewNN.Close();
    
        isFollowObject = false;
        panel.SetActive(false);
    }

}
