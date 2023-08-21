using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;


using SkillsBaseAndVal = Pair<float, float>;


public class Genome
{
    public NN actNN;
    public NN rewNN;

    public Epoch actNNepoch;
    public Epoch rewNNepoch;

    public Dictionary<string, SkillsBaseAndVal> skills;

    public float r, g, b;

    public Genome(int inputSize, int outputSize)
    {
        // SETTEING DATA
        actNN = new NN(inputSize, 10, outputSize);
        actNN.learningRate = 0.01;
        actNN.InitRandom();
        
        rewNN = new NN(actNN.sizes.Sum(), 5, 2);
        rewNN.learningRate = 0.001;
        rewNN.InitRandom();
        rewNN.SetActivation(relu);
        rewNN.SetDerivative(relu_der);

        r = UnityEngine.Random.Range(0f, 1f);
        g = UnityEngine.Random.Range(0f, 1f);
        b = UnityEngine.Random.Range(0f, 1f);

        skills = new Dictionary<string, SkillsBaseAndVal>()
        {
            // SETTEING DATA
            ["speed"] = new SkillsBaseAndVal(1f, 0.7f),
            ["angularSpeed"] = new SkillsBaseAndVal(1f, 10f),
            ["foodAspect"] = new SkillsBaseAndVal(0.75f, 1f), // non-live food assimilation like green circles
            ["memoryFactor"] = new SkillsBaseAndVal(0.6f, 1f), //
            //["size"] = new SkillsBaseAndVal(1f, 0.5f),
            ["needEnergyDivide"] = new SkillsBaseAndVal(15f*3, Config.stepsPerEpoch),
        };

        actNNepoch = new Epoch(Config.stepsPerEpoch);
        rewNNepoch = new Epoch(Config.stepsPerEpoch);
        rewNNepoch.roundFunc =       x => x < 0.5 ? 0d : 1d;
        rewNNepoch.invertRoundFunc = x => x > 0.5 ? 0d : 1d;
    }

    public Genome(Genome refGenome)
    {
        actNN = new NN(refGenome.actNN);
        rewNN = new NN(refGenome.rewNN);

        r = refGenome.r;
        g = refGenome.g;
        b = refGenome.b;

        skills = new Dictionary<string, SkillsBaseAndVal>();
        foreach (var entry in refGenome.skills)
        {
            var val = new SkillsBaseAndVal(entry.Value.First, entry.Value.Second);
            skills.Add(entry.Key, val);
        }

        actNNepoch = new Epoch(refGenome.actNNepoch);
        rewNNepoch = new Epoch(refGenome.rewNNepoch);
    }

    public void Cross(Genome gen)
    {
        actNN.Cross(gen.actNN);
        rewNN.Cross(gen.rewNN);

        r = (r + gen.r) / 2;
        g = (g + gen.g) / 2;
        b = (b + gen.b) / 2;

        List<string> keyList = new List<string>(skills.Keys);
        foreach (string entry in keyList)
        {
            if (UnityEngine.Random.value < 0.5)
                skills[entry] = gen.skills[entry];
        }
    }

    public float GetActualSkill(string name)
    {
        return skills[name].First * skills[name].Second;
    }

    public void Mutate(float weightMutAmpl, float weightProb, float skillMutAmpl, float skillProb)
    {
        for (int i = 0; i < rewNN.layers.Length; i++)
        {
            // SETTEING DATA
            rewNN.layers[i].mutateLayer(0.07, 0.03);
        }
        
        for (int i = 0; i < actNN.layers.Length; i++)
        {
            actNN.layers[i].mutateLayer(weightMutAmpl, weightProb);
        }

        foreach (var skill in skills)
        {
            if (UnityEngine.Random.value < skillProb)
                skills[skill.Key].First += UnityEngine.Random.Range(-skillMutAmpl, skillMutAmpl);
        }

        skills["foodAspect"].First = Mathf.Clamp01(skills["foodAspect"].First);
        //skills["size"].First = Mathf.Clamp(skills["size"].First, 0.5f, 3f);
        skills["memoryFactor"].First = Mathf.Clamp(skills["memoryFactor"].First, 0.05f, 1f);
        skills["speed"].First = Mathf.Clamp(skills["speed"].First, 0.1f, 1.1f);
        //skills["speed"].First = 1f;

        float colorD = 15f / 225f;
        float colorProb = 0.6f;
        if (UnityEngine.Random.value < colorProb) r = Mathf.Clamp01(r + UnityEngine.Random.Range(-colorD, colorD));
        if (UnityEngine.Random.value < colorProb) g = Mathf.Clamp01(g + UnityEngine.Random.Range(-colorD, colorD));
        if (UnityEngine.Random.value < colorProb) b = Mathf.Clamp01(b + UnityEngine.Random.Range(-colorD, colorD));
    }

    public static double sigmoid(double value)
    {
        return 1d / (1d + (double)Math.Exp(-value));
    }
    
    public static double sigmoid_der(double value)
    {
        return sigmoid(value) * (1 - sigmoid(value));
    }
    
    public static double relu(double value)
    {
        if (value < 0) return value * 0.01d;
        else if (value > 1) return (value - 1d) * 0.01d + 1d;
        else return value;
    }

    public static double relu_der(double value)
    {
        if (value < 0 || value > 1) return 0.01;
        else return 1;
    }
}
