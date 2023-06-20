using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using UnityEngine;


using SkillsBaseAndVal = Pair<float, float>;


public class Genome
{
    public NN nn;

    public Dictionary<string, SkillsBaseAndVal> skills;

    public float r, g, b;

    public Genome(int inputSize, int outputSize)
    {
        nn = new NN(inputSize, 5, outputSize);

        r = UnityEngine.Random.Range(0f, 1f);
        g = UnityEngine.Random.Range(0f, 1f);
        b = UnityEngine.Random.Range(0f, 1f);

        skills = new Dictionary<string, SkillsBaseAndVal>()
        {
            ["speed"] = new SkillsBaseAndVal(1f, 7.5f/20),
            ["angularSpeed"] = new SkillsBaseAndVal(1f, 160f/20),
            //["foodAspect"] = new SkillsBaseAndVal(1f, 1f),
            ["memoryFactor"] = new SkillsBaseAndVal(0.5f, 1f), //
            //["size"] = new SkillsBaseAndVal(1f, 0.5f),
            ["needEnergyDivide"] = new SkillsBaseAndVal(1f, 16f*20),
        };
    }

    public Genome(Genome refGenome)
    {
        nn = new NN(refGenome.nn);

        r = refGenome.r;
        g = refGenome.g;
        b = refGenome.b;

        skills = new Dictionary<string, SkillsBaseAndVal>(refGenome.skills);
    }

    public float GetActualSkill(string name)
    {
        return skills[name].First * skills[name].Second;
    }

    public void Mutate(float weightMutAmpl, float weightProb, float skillMutAmpl, float skillProb)
    {
        for (int i = 0; i < nn.layers.Length; i++)
        {
            nn.layers[i].mutateLayer(weightMutAmpl, weightProb);
        }

        foreach (var skill in skills)
        {
            if (UnityEngine.Random.value < skillProb)
                skills[skill.Key].First += UnityEngine.Random.Range(-skillMutAmpl, skillMutAmpl);
        }

        //skills["foodAspect"].First = Mathf.Clamp01(skills["foodAspect"].First);
        //skills["size"].First = Mathf.Clamp(skills["size"].First, 0.5f, 3f);
        skills["memoryFactor"].First = Mathf.Clamp(skills["memoryFactor"].First, 0.05f, 1f);

        float colorD = 7f / 225f;
        float colorProb = 0.6f;
        if (UnityEngine.Random.value < colorProb) r = Mathf.Clamp01(r + UnityEngine.Random.Range(-colorD, colorD));
        if (UnityEngine.Random.value < colorProb) g = Mathf.Clamp01(g + UnityEngine.Random.Range(-colorD, colorD));
        if (UnityEngine.Random.value < colorProb) b = Mathf.Clamp01(b + UnityEngine.Random.Range(-colorD, colorD));
    }
}
