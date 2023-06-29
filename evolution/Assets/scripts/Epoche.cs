using Accord.Math;
using Accord.Statistics;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;


public class Epoch
{
    public int size;
    
    public double[] rewards;
    public double[] punishs;
    public double[][] inputs;
    public double[][] outputs;

    public double gamma = 0.88;

    private int epochI = 0;

    public Epoch(int size) 
    { 
        this.size = size;
        rewards = new double[size];
        punishs = new double[size];
        inputs = new double[size][];
        outputs = new double[size][];
    }

    public void AddEpoch(double[] input, double[] output, double reward, double punish)
    {
        if (epochI != 0)
        {
            UnityEngine.Debug.Assert(input.Length == inputs[0].Length);
            UnityEngine.Debug.Assert(output.Length == outputs[0].Length);
        }

        inputs[epochI] = new double[input.Length];
        Array.Copy(input, inputs[epochI], input.Length);
        
        outputs[epochI] = new double[output.Length];
        Array.Copy(output, outputs[epochI], output.Length);

        rewards[epochI] = reward;
        punishs[epochI] = punish;
        
        epochI++;
    }

    public bool IsDone()
    {
        return epochI >= size;
    }

    public void Apply(ref NN nn)
    {
        UnityEngine.Debug.Log("Reward summ: " + rewards.Sum().ToString("0.0000"));
        UnityEngine.Debug.Log("Punishment summ: " + punishs.Sum().ToString("0.0000"));

        epochI = 0;

        var outs = Matrix.ToMatrix(outputs);

        double[,] explicitOuts = ToExplicitOutput(outs);

        double[] discountedReward = NormGrade(rewards);
        var rewMap = discountedReward.Outer(Accord.Math.Vector.Ones(outs.GetLength(1)));
        
        double[] discountedPuns = NormGrade(punishs);
        var punMap = discountedPuns.Outer(Accord.Math.Vector.Ones(outs.GetLength(1)));

        double[,] dersR = explicitOuts.Subtract(outs).Multiply(rewMap);
        double[,] dersP = explicitOuts.Multiply(-1d).Subtract(outs).Multiply(punMap);

        double[,] ders = dersR.Add(dersP);

        nn.BackProp(inputs, ders.ToJagged());
    }

    double[,] ToExplicitOutput(double[,] outputs)
    {
        double[,] outp = Matrix.Copy(outputs);
        outp = outp.Apply(x => x < 0d ? -1d : 1d);

        return outp;
    }

    double[] NormGrade(double[] grade)
    {
        double[] newGrades = Accord.Math.Vector.Zeros(grade.Length);

        // Spreading reward for the previous actions
        double runningAdd = 0;
        for (int i = size - 1; i >= 0; i--)
        {
            if (Math.Abs(grade[i]) >= Math.Abs(runningAdd) || ((grade[i] < 0) != (runningAdd < 0))) 
                runningAdd = 0;
            runningAdd = runningAdd * gamma + grade[i];
            newGrades[i] = runningAdd;
        }

        double mean = newGrades.Mean();
        //double dev = StandardDeviation(newGrades);

        newGrades = newGrades.Subtract(mean);
        
        return newGrades;
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
