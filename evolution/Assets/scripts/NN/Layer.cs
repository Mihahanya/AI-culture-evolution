using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Numpy;

public class Layer
{
    public int inputSize { get; }
    public int outSize { get; }
    public NDarray weights;
    public NDarray biases;
    public NDarray neurons;

    public Layer(int inputSize, int outSize)
    {
        this.outSize = outSize;
        this.inputSize = inputSize;
        neurons = np.zeros(outSize);
        weights = np.zeros(outSize, inputSize);
        biases = np.zeros(outSize);
    }

    public Layer(Layer l) : this(l.inputSize, l.outSize)
    {
        weights = np.copy(l.weights);
        biases = np.copy(l.biases);
    }

    public void calcLayer(NDarray inputs)
    {
        neurons = np.tanh(np.dot(weights, inputs) + biases);
    }

    public NDarray calcLayerVals(NDarray inputs)
    {
        calcLayer(inputs);
        return neurons;
    }

    public void initRandom(float min, float max)
    {
        //weights = np.random.randn(outSize, inputSize);
        //biases = np.random.randn(outSize);
        weights = np.random.rand(outSize, inputSize) * 2f - 1f;
        biases = np.random.rand(outSize) * 2f - 1f;
    }

    public void mutateLayer(float mutationAmplitude, float probability)
    {
        var mask = np.random.choice(
            new float[] { 1f, 0f }, 
            new int[] {inputSize, outSize}, 
            true, 
            new float[] {probability, 1f - probability}
        );
        weights += mask * (np.random.rand(inputSize, outSize) * 2f - 1f) * mutationAmplitude;
        
        var biasMask = np.random.choice(
            new float[] { 1f, 0f }, 
            new int[] {inputSize}, 
            true, 
            new float[] {probability, 1f - probability}
        );
        biases += biasMask * (np.random.rand(inputSize) * 2f - 1f) * mutationAmplitude;
    }
}
