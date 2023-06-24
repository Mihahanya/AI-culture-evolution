using Accord.Math;
using Accord.Statistics;
using System;
using System.Linq;
using System.Numerics;


public class Epoch
{
    public int size;
    
    public double[] rewards;
    public double[][] inputs;
    public double[][] outputs;

    public double gamma = 0.9;

    private int epochI = 0;

    public Epoch(int size) 
    { 
        this.size = size;
        rewards = new double[size];
        inputs = new double[size][];
        outputs = new double[size][];
    }

    public void AddEpoch(double[] input, double[] output, double reward)
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
        UnityEngine.Debug.Log("Reward summ: " + rewards.Sum());

        var outs = Matrix.ToMatrix(outputs);

        double[,] explicitOuts = ToExplicitOutput(outs);

        double[,] des = explicitOuts.Subtract(outs);

        double[] discountedReward = NormRewards(rewards);
        var rewMap = discountedReward.Outer(Accord.Math.Vector.Ones(des.GetLength(1)));
        //des = des.Transpose().Dot(rewMatrix).Transpose();

        des = des.Multiply(rewMap);

        nn.BackProp(inputs, des.ToJagged());

        epochI = 0;
    }

    double[,] ToExplicitOutput(double[,] outputs)
    {
        double[,] outp = Matrix.Copy(outputs);
        outp = outp.Apply(x => x < 0d ? -1d : 1d);

        return outp;
    }

    double[] NormRewards(double[] rews)
    {
        double[] newRews = Accord.Math.Vector.Zeros(rews.Length);

        // Spreading reward for the previous actions
        double runningAdd = 0;
        for (int i = size - 1; i >= 0; i--)
        {
            if (rews[i] != 0) runningAdd = 0;
            runningAdd = runningAdd * gamma + rews[i];
            newRews[i] = runningAdd;
        }

        double mean = newRews.Mean();
        //double dev = StandardDeviation(newRews);

        newRews = newRews.Subtract(mean);
        
        return newRews;
    }

    double StandardDeviation(double[] a)
    {
        double mean = a.Average();

        double res = 0;
        
        for (int i = 0; i < a.Length; i++) 
            res += (double)Math.Pow(a[i] - mean, 2f) / a.Length;

        return (double)Math.Sqrt(res);
    }
}
