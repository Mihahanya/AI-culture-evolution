using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using Accord.Math.Random;

public class LayerData
{
    public int inputSize { get; }
    public int outSize { get; }
    public double[,] weights;
    public double[] biases;

    public LayerData(int inputSize, int outSize)
    {
        this.outSize = outSize;
        this.inputSize = inputSize;
        weights = Matrix.Zeros(outSize, inputSize);
        biases = Vector.Zeros(outSize);
    }

    public LayerData(LayerData l)
    {
        inputSize = l.inputSize;
        outSize = l.outSize;

        weights = l.weights.Clone() as double[,];

        biases = new double[l.biases.Length];
        Array.Copy(l.biases, biases, l.biases.Length);
    }
}

public class Layer : LayerData
{
    public Func<double, double> activation = Math.Tanh;
    public Func<double, double> derivative = tanh_der;

    public double[] neurons;

    public Layer(int inputSize, int outSize) : base(inputSize, outSize)
    {
        neurons = Vector.Zeros(outSize);
    }

    public Layer(Layer l) : base(l) 
    {
        activation = l.activation;
        derivative = l.derivative;
    }

    public void calcLayer(double[] inputs)
    {
        Debug.Assert(inputs.Length == inputSize);

        neurons = Matrix.Dot(weights, inputs);
        neurons = neurons.Add(biases);
        neurons = Matrix.Apply(neurons, activation);
    }

    public double[] calcLayerVals(double[] inputs)
    {
        calcLayer(inputs);
        return neurons;
    }

    public void InitRandom()
    {
        var normalRand = new ZigguratNormalGenerator();

        weights = Matrix.Random(outSize, inputSize, normalRand).Divide(3d);
        biases = normalRand.Generate(outSize).Divide(3d);
    }

    public void Reset()
    {
        neurons = Vector.Zeros(outSize);
        weights = Matrix.Zeros(outSize, inputSize);
        biases = Vector.Zeros(outSize);
    }

    public void mutateLayer(double mutationAmplitude, double probability)
    {
        Func<double, double> fillAndInterpolate = (x) => UnityEngine.Random.value < probability ? (x * 2d - 1d) * mutationAmplitude : 0d;

        var weightsMask = Matrix.Random(outSize, inputSize);
        weightsMask = weightsMask.Apply(fillAndInterpolate);

        var biasMask = Vector.Random(outSize);
        biasMask = biasMask.Apply(fillAndInterpolate);

        weights = weights.Add(weightsMask);
        biases = biases.Add(biasMask);
    }

    //public void mutateLayer(float mutationAmplitude, float probability)
    //{
    //    for (int i = 0; i < outSize; i++)
    //    {
    //        if (UnityEngine.Random.value < probability)
    //            biases[i] += UnityEngine.Random.Range(-mutationAmplitude, mutationAmplitude);
    //        for (int j = 0; j < inputSize; j++)
    //        {
    //            if (UnityEngine.Random.value < probability)
    //                weights[i, j] += UnityEngine.Random.Range(-mutationAmplitude, mutationAmplitude);
    //        }
    //    }
    //}

    public static double tanh_der(double x)
    {
        return 1d - (double)Math.Pow(Math.Tanh(x), 2d);
    }
}
