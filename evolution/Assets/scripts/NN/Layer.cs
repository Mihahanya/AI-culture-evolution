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
    
    public double[] neurons;

    public Layer(int inputSize, int outSize) : base(inputSize, outSize)
    {
        neurons = Vector.Zeros(outSize);
    }

    public Layer(Layer l) : base(l) { }

    public void calcLayer(double[] inputs)
    {
        Debug.Assert(inputs.Length == inputSize);

        neurons = Matrix.Dot(weights, inputs);
        neurons = neurons.Add(biases);
        neurons = Matrix.Apply(neurons, Math.Tanh);
    }

    public double[] calcLayerVals(double[] inputs)
    {
        calcLayer(inputs);
        return neurons;
    }

    public void initRandom(double min, double max)
    {
        weights = Matrix.Random(outSize, inputSize).Multiply(2).Add(-1).Multiply(max);
        biases = Vector.Random(outSize).Multiply(2).Add(-1).Multiply(max);
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
}
