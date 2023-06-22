using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEditor.PackageManager;
using UnityEngine;
using Numpy;


public class Epoch
{
    public int size;
    
    public float[] rewards;
    public NDarray inputs;
    public NDarray outputs;

    private int epochI = 0;

    public Epoch(int size) 
    { 
        this.size = size;
        rewards = new float[size];
    }

    public void AddEpoch(NDarray input, NDarray output, float reward)
    {
        if (epochI == 0)
        {
            inputs = np.zeros(size, input.len);
            outputs = np.zeros(size, output.len);
        }

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
        UnityEngine.Debug.Log("Reward summ: " + rewards.Sum());

        NDarray discountedReward = NormRewards(rewards);
        NDarray explicitOuts = ToExplicitOutput(outputs);
        NDarray des = ((explicitOuts - outputs).T * discountedReward).T * 0.002f;

        //for (int i = 0; i < size; i++)
        //{
        //    NDarray outs = outputs[i];
        //    NDarray explicitOuts = ToExplicitOutput(outs);
        //
        //    for (int j = 0; j < outs.len; j++)
        //    {
        //        des[i][j] = (NDarray)((explicitOuts[j] - outs[j]) * discountedReward[i] * 0.002f);
        //    }
        //}

        nn.BackProp(inputs, des);

        epochI = 0;
    }

    NDarray ToExplicitOutput(NDarray outputs)
    {
        NDarray outp = np.copy(outputs);
        outp[outp < 0] = (NDarray)(-1);
        outp[outp >= 0] = (NDarray)(1);

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
