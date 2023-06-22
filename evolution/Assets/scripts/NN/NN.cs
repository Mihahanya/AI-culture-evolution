using Numpy;
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
        Array.Copy(nn.layers, layers, nn.layers.Length);
    }

    public NDarray FeedForward(NDarray inputs)
    {
        layers[0].calcLayer(inputs);

        for (int i = 1; i < layers.Length; i++)
        {
            layers[i].calcLayer(layers[i-1].neurons);
        }

        return layers[layers.Length-1].neurons;
    }

    public NDarray ExplicitOutput()
    {
        NDarray outp = np.copy(layers[layers.Length - 1].neurons);
        outp[outp < 0] = (NDarray)(-1);
        outp[outp >= 0] = (NDarray)(1);

        return outp;
    }

    public void BackProp(NDarray inputs, NDarray errors)
    {
        return;

        Layer[] newLayers = new Layer[layers.Length];
        Array.Copy(layers, newLayers, layers.Length);
        //for (int i = 0; i < sizes.Length; i++)
        //    newLayers[i] = new Layer(layers[i]);

        for (int t = 0; t < errors.len; t++) // go through batch
        { 
            var input = inputs[t];
            var error = errors[t];

            var _ = FeedForward(input); // to take neurons states

            for (int k = layers.Length - 1; k >= 0; k--) // through layers
            {
                Layer l = layers[k];

                NDarray prevNeurons;
                if (k != 0) prevNeurons = layers[k - 1].neurons;
                else prevNeurons = input;

                // Update weights

                var wGradients = np.outer(error, prevNeurons);

                var bGradients = error;

                newLayers[k].weights += wGradients;
                newLayers[k].biases += bGradients;

                // Errors for the next layers
                
                error = np.dot(l.weights.T, error);
            }
        }

        //for (int i = 0; i <  layers.Length; i++)
        //{
        //
        //}

        Array.Copy(newLayers, layers, newLayers.Length);
    }

    float arctanh(float x)
    {
        return 1f - (float)Math.Pow(Math.Tanh(x), 2f);
    }

}

