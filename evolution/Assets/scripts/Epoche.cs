using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEditor.PackageManager;
using UnityEngine;


public class Epoch
{
    public int size;
    
    public float[] rewards;
    public float[][] inputs;
    public float[][] outputs;

    private int epochI = 0;

    public Epoch(int size) 
    { 
        this.size = size;
        inputs = new float[size][];
        outputs = new float[size][];
        rewards = new float[size];
    }

    public void AddEpoch(float[] input, float[] output, float reward)
    {
        inputs[epochI] = input;
        outputs[epochI] = output;
        rewards[epochI] = reward;
        epochI++;
    }

    public bool IsDone()
    {
        return epochI >= size;
    }

    public void Apply(ref NN nn)
    {
        float[][] des = new float[size][];

        UnityEngine.Debug.Log("Reward summ: " + rewards.Sum());

        float[] discountedReward = NormRewards(rewards);
        
        for (int i = 0; i < size; i++)
        {
            float[] outs = outputs[i];
            float[] explicitOuts = ToExplicitOutput(outs);

            des[i] = new float[outs.Length];
            for (int j = 0; j < outs.Length; j++)
            {
                des[i][j] = (explicitOuts[j] - outs[j]) * discountedReward[i] * 0.002f;
            }
        }

        nn.BackProp(inputs, des);

        epochI = 0;
    }

    float[] ToExplicitOutput(float[] outputs)
    {
        float[] outp = new float[outputs.Length];

        for (int i = 0; i < outputs.Length; i++)
        {
            if (outputs[i] < 0f) outp[i] = -1f;
            else outp[i] = 1f;
        }

        return outp;
    }

    float[] NormRewards(float[] rews)
    {
        float[] newRews = new float[rews.Length];

        // Spreading reward for the previous actions
        float runningAdd = 0;
        for (int i = size - 1; i >= 0; i--)
        {
            if (rews[i] != 0) runningAdd = 0;
            runningAdd = runningAdd * 0.9f + rews[i];
            newRews[i] = runningAdd;
        }

        float mean = newRews.Average();
        float dev = StandardDeviation(newRews);

        for (int i = 0; i < size; i++)
        {
            newRews[i] -= mean;
            //normRewards[i] /= mean / 2f;
            //normRewards[i] /= dev;
        }

        return newRews;
    }

    float StandardDeviation(float[] a)
    {
        float mean = a.Average();

        float res = 0;
        
        for (int i = 0; i < a.Length; i++) 
            res += (float)Math.Pow(a[i] - mean, 2f) / (float)a.Length;

        return (float)Math.Sqrt(res);
    }
}
