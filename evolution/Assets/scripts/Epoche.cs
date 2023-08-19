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
    public double[][][] states; // batch x layer x val

    public double gamma = 0.88;

    private int epochI = 0;

    public Epoch(int size)
    {
        this.size = size;
        rewards = new double[size];
        punishs = new double[size];
        states = new double[size][][];
    }

    public void AddEpoch(double[] input, double[][] state, double reward, double punish)
    {
        if (epochI != 0)
        {
            UnityEngine.Debug.Assert(state.Length + 1 == states[0].Length); // layers number
            // check every layer size
            UnityEngine.Debug.Assert(input.Length == states[0][0].Length);
            for (int i = 0; i < state.Length; i++)
                UnityEngine.Debug.Assert(state[i].Length == states[0][i + 1].Length);
        }

        states[epochI] = new double[state.Length + 1][];
        states[epochI][0] = new double[input.Length];
        Array.Copy(input, states[epochI][0], input.Length);
        Array.Copy(state, 0, states[epochI], 1, state.Length);

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
        epochI = 0;

        var statesTrans = states.Transpose();

        double[,] outs = statesTrans[statesTrans.Length-1].ToMatrix();

        double[,] explicitOuts = ToExplicitOutput(outs);

        double[] discountedReward = NormGrade(rewards);
        double[] discountedPuns = NormGrade(punishs);
        
        double[,] dersR = explicitOuts.Subtract(outs).TransposeAndDotWithDiagonal(discountedReward).Transpose();
        double[,] dersP = explicitOuts.Multiply(-1d).Subtract(outs).TransposeAndDotWithDiagonal(discountedPuns).Transpose();

        double[,] ders = dersR.Add(dersP);

        nn.BackProp(statesTrans, ders);
        //nn.BackProp(states.GetColumn(0), ders.ToJagged());
    }

    double[,] ToExplicitOutput(double[,] outputs)
    {
        return outputs.Apply(x => x < 0d ? -1d : 1d);
    }

    double[][] ToExplicitOutput(double[][] outputs)
    {
        return outputs.Apply(x => x.Apply(y => y < 0d ? -1d : 1d));
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

        //newGrades = newGrades.Subtract(mean);

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
