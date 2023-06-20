using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class NN
{
    public Layer[] layers;
    public int[] sizes;

    public NN(params int[] sizes)
    {
        this.sizes = sizes;

        layers = new Layer[sizes.Length-1];
        for (int i = 1; i < sizes.Length; i++)
        {
            layers[i-1] = new Layer(sizes[i-1], sizes[i]);
            layers[i-1].initRandom(-1f, 1f);
        }
    }

    public NN(NN nn)
    {
        sizes = nn.sizes;
        layers = new Layer[nn.layers.Length];

        for (int i = 0; i < nn.layers.Length; i++)
            layers[i] = new Layer(nn.layers[i]);
    }

    public float[] FeedForward(float[] inputs)
    {
        layers[0].calcLayer(inputs);

        for (int i = 1; i < layers.Length; i++)
        {
            layers[i].calcLayer(layers[i-1].neurons);
        }

        return layers[layers.Length-1].neurons;
    }

    public float[] ExplicitOutput()
    {
        float[] lastLayerNeurons = layers[layers.Length - 1].neurons;
        float[] outp = new float[lastLayerNeurons.Length];

        for (int i = 0; i < lastLayerNeurons.Length;i++)
        {
            float outval = lastLayerNeurons[i];
            //if (outval <= -1f / 6f) outp[i] = -1;
            //else if (outval >= 1f / 6f) outp[i] = 1;
            //else outp[i] = 0;

            if (outval < 0f) outp[i] = -1f;
            else outp[i] = 1f;
        }

        return outp;
    }

    public void BackProp(float[][] inputs, float[][] errors)
    {
        Layer[] newLayers = new Layer[layers.Length];
        Array.Copy(layers, newLayers, layers.Length);
        //for (int i = 0; i < sizes.Length; i++)
        //    newLayers[i] = new Layer(layers[i]);

        for (int t = 0; t < errors.Length; t++) // go through batch
        { 
            float[] input = inputs[t];
            float[] error = errors[t];

            float[] _ = FeedForward(input); // to take neurons states

            for (int k = layers.Length - 1; k >= 0; k--) // through layers
            {
                Layer l = layers[k];

                float[] prevNeurons;
                if (k != 0) prevNeurons = layers[k - 1].neurons;
                else prevNeurons = input;

                // Update weights
                for (int i = 0; i < l.outSize; i++)
                {
                    for (int j = 0; j < l.inputSize; j++)
                    {
                        newLayers[k].weights[i, j] += prevNeurons[j] * error[i] * arctanh(l.neurons[i]);
                    }
                    newLayers[k].biases[i] += error[i] * arctanh(l.neurons[i]);
                }

                // Errors for the next layers
                float[] nextErrors = new float[l.inputSize];
                for (int i = 0; i < l.inputSize; i++)
                {
                    for (int j = 0; j < l.outSize; j++)
                    {
                        nextErrors[i] += error[j] * l.weights[j, i];
                    }
                }
                error = new float[l.inputSize];
                Array.Copy(nextErrors, error, nextErrors.Length);
            }
        }

        Array.Copy(newLayers, layers, newLayers.Length);
    }

    float arctanh(float x)
    {
        return 1f - (float)Math.Pow(Math.Tanh(x), 2f);
    }

}

