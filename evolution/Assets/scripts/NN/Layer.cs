using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public int outSize, inputSize;
    public float[,] weights;
    public float[] biases;
    public float[] neurons;

    public Layer(int inputSize, int outSize)
    {
        this.outSize = outSize;
        this.inputSize = inputSize;
        neurons = new float[outSize];
        weights = new float[outSize, inputSize];
        biases = new float[outSize];
    }

    public Layer(Layer l) : this(l.inputSize, l.outSize)
    {
        for (int i = 0; i < outSize; i++) {
            biases[i] = l.biases[i];

            for (int j = 0; j < inputSize; j++)
                weights[i, j] = l.weights[i, j];
        }
    }

    public void calcLayer(float[] inputs)
    {
        Debug.Assert(inputs.Length == inputSize);

        for (int i = 0; i < outSize; i++)
        {
            neurons[i] = biases[i];

            for (int j = 0; j < inputSize; j++)
                neurons[i] += inputs[j] * weights[i, j];
            
            neurons[i] = (float)Math.Tanh(neurons[i]);
        }
    }

    public float[] calcLayerVals(float[] inputs)
    {
        calcLayer(inputs);
        return neurons;
    }

    public void initRandom(float min, float max)
    {
        for (int i = 0; i < outSize; i++)
        {
            biases[i] = UnityEngine.Random.Range(min, max);

            for (int j = 0; j < inputSize; j++)
                weights[i, j] = UnityEngine.Random.Range(min, max);
            
        }
    }

    public void mutateLayer(float mutationAmplitude, float probability)
    {
        for (int i = 0; i < outSize; i++)
        {
            if (UnityEngine.Random.value < probability)
                biases[i] += UnityEngine.Random.Range(-mutationAmplitude, mutationAmplitude);

            for (int j = 0; j < inputSize; j++)
            {
                if (UnityEngine.Random.value < probability) 
                    weights[i, j] += UnityEngine.Random.Range(-mutationAmplitude, mutationAmplitude);
            }
        }
    }
}
