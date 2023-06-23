using Accord.Math;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class NN
{
    public Layer[] layers;
    private Layer[] RMSpropCache;
    public int[] sizes;
    public double learningRate = 0.001f;
    public double decayRate = 0.99;

    public NN(params int[] sizes)
    {
        this.sizes = sizes;

        layers = new Layer[sizes.Length-1];
        RMSpropCache = new Layer[layers.Length];
        for (int i = 1; i < sizes.Length; i++)
        {
            layers[i-1] = new Layer(sizes[i-1], sizes[i]);
            layers[i-1].initRandom(-1f, 1f);
            
            RMSpropCache[i-1] = new Layer(sizes[i-1], sizes[i]);
            RMSpropCache[i-1].Reset();
        }
    }

    public NN(NN nn)
    {
        sizes = nn.sizes;
        layers = new Layer[nn.layers.Length];
        Array.Copy(nn.layers, layers, nn.layers.Length);

        RMSpropCache = new Layer[layers.Length];
        for (int i = 1; i < sizes.Length; i++)
        {
            RMSpropCache[i-1] = new Layer(sizes[i-1], sizes[i]);
            RMSpropCache[i-1].Reset();
        }
    }

    public double[] FeedForward(double[] inputs)
    {
        layers[0].calcLayer(inputs);

        for (int i = 1; i < layers.Length; i++)
        {
            layers[i].calcLayer(layers[i-1].neurons);
        }

        return layers[layers.Length-1].neurons;
    }

    public double[] ExplicitOutput()
    {
        double[] outp = Vector.Copy(layers[layers.Length - 1].neurons);
        outp = outp.Apply(x => x < 0d ? -1d : 1d);

        return outp;
    }

    public void BackProp(double[][] inputs, double[][] errors)
    {
        Layer[] gradBuff = new Layer[layers.Length];
        Array.Copy(layers, gradBuff, layers.Length);
        
        for (int t = 0; t < errors.Length; t++) // go through batch
        { 
            var input = inputs[t];
            var error = errors[t];

            var _ = FeedForward(input); // to take neurons states

            for (int k = layers.Length - 1; k >= 0; k--) // through layers
            {
                Layer l = layers[k];

                double[] prevNeurons;
                if (k != 0) prevNeurons = layers[k - 1].neurons;
                else prevNeurons = input;

                // Update weights

                var ders = l.neurons.Apply(arctanh);
                var wGradients = error.Multiply(ders).Outer(prevNeurons);

                var bGradients = error;

                gradBuff[k].weights = gradBuff[k].weights.Add(wGradients);
                gradBuff[k].biases = gradBuff[k].biases.Add(bGradients);

                // Errors for the next layers
                
                error = Matrix.Dot(l.weights.Transpose(), error);
            }
        }

        for (int i = 0; i < layers.Length; i++)
        {
            double[,] g = Matrix.Copy(gradBuff[i].weights);
            //var gSqr = g.Apply(x => x * x);
            //RMSpropCache[i].weights = RMSpropCache[i].weights.Multiply(decayRate).Add(gSqr.Multiply(1d - decayRate));
            //var rmspropsqrt = RMSpropCache[i].weights.Apply(Math.Sqrt);
            //layers[i].weights = layers[i].weights.Add(g.Multiply(learningRate).Divide(rmspropsqrt.Add(1e-5)));

            layers[i].weights = layers[i].weights.Add(gradBuff[i].weights.Multiply(learningRate));
            layers[i].biases = layers[i].biases.Add(gradBuff[i].biases.Multiply(learningRate));
        }

        //Array.Copy(newLayers, layers, newLayers.Length);
    }

    double arctanh(double x)
    {
        return 1f - (float)Math.Pow(Math.Tanh(x), 2f);
    }

}

