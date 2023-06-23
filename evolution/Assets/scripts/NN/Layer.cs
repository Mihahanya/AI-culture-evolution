using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using Accord.Math.Random;


public class Layer
{
    public int inputSize { get; }
    public int outSize { get; }
    public double[,] weights;
    public double[] biases;
    public double[] neurons;

    public Layer(int inputSize, int outSize)
    {
        this.outSize = outSize;
        this.inputSize = inputSize;
        neurons = Vector.Zeros(outSize);
        weights = Matrix.Zeros(outSize, inputSize);
        biases = Vector.Zeros(outSize);
    }

    public Layer(Layer l) : this(l.inputSize, l.outSize)
    {
        weights = Matrix.Copy(l.weights);
        biases = Vector.Copy(l.biases);
    }

    public void calcLayer(double[] inputs)
    {
        neurons = Vector.Zeros(outSize);
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
        Func<double, double> fillAndInterpolate = (x) => UnityEngine.Random.value < probability ? (x * 2 - 1) * mutationAmplitude : 0;

        var weightsMask = Matrix.Random(inputSize, outSize);
        weightsMask = weightsMask.Apply(fillAndInterpolate);

        var biasMask = Vector.Random(outSize);
        biasMask = biasMask.Apply(fillAndInterpolate);

        weights = weights.Add(weightsMask);
        biases = biases.Add(biasMask);
    }
}
