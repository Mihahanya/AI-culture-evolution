using Accord.Math;
using Accord.Statistics;
using Accord.Statistics.Kernels;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Resources;
using System.Runtime.InteropServices.ComTypes;

public class Epoch
{
    public int size;
    
    public double[] rewards;
    public double[] punishs;
    public double[,][] states; // batch x layer x val
    public double[][] outputs; 

    public double gamma = 0.88;

    private int epochI = 0;

    public Epoch(int size) 
    { 
        this.size = size;
        rewards = new double[size];
        punishs = new double[size];
        outputs = new double[size][];
        //states = new double[size][][];
    }

    public void AddEpoch(double[] input, double[][] state, double reward, double punish)
    {
        if (epochI != 0)
        {
            UnityEngine.Debug.Assert(state.Length+1 == states.GetLength(1));
            UnityEngine.Debug.Assert(input.Length == states[0, 0].Length);
            for (int i = 0; i < state.Length; i++) 
                UnityEngine.Debug.Assert(state[i].Length == states[0, i + 1].Length);
        }
        else
        {
            states = new double[size, state.Length + 1][];
        }

        states[epochI, 0] = new double[input.Length];
        Array.Copy(input, states[epochI, 0], input.Length);

        for (int i = 0; i < state.Length; i++)
        {
            states[epochI, i+1] = new double[state[i].Length];
            Array.Copy(state[i], states[epochI, i + 1], state[i].Length);
        }

        outputs[epochI] = state[state.Length-1];

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

        //var outs = states.GetColumn(states.GetLength(1)-1).ToMatrix();
        //int outSize = states[0, states.GetLength(1) - 1].Length;
        //var outs = new double[size, outSize];
        //for (int i = 0; i < size; i++)
        //{
        //    for (int j = 0; j < outSize; j++)
        //        outs[i, j] = states[i, states.GetLength(1)-1][j];
        //
        //}

        var outs = outputs.ToMatrix();

        double[,] explicitOuts = ToExplicitOutput(outs);

        double[] discountedReward = NormGrade(rewards);
        var rewMap = discountedReward.Outer(Accord.Math.Vector.Ones(outs.GetLength(1)));
        
        double[] discountedPuns = NormGrade(punishs);
        var punMap = discountedPuns.Outer(Accord.Math.Vector.Ones(outs.GetLength(1)));

        double[,] dersR = explicitOuts.Subtract(outs).Multiply(rewMap);
        double[,] dersP = explicitOuts.Multiply(-1d).Subtract(outs).Multiply(punMap);

        double[,] ders = dersR.Add(dersP);

        nn.BackProp(states.Transpose().ToJagged(), ders.ToJagged());
        //nn.BackProp(states.GetColumn(0), ders.ToJagged());
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

    static int[][] CopyArray(int[][] source)
    {
        var len = source.Length;
        var dest = new int[len][];

        for (var x = 0; x < len; x++)
        {
            var inner = source[x];
            var ilen = inner.Length;
            var newer = new int[ilen];
            Array.Copy(inner, newer, ilen);
            dest[x] = newer;
        }

        return dest;
    }
}
