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
    public double learningRate = 0.005f;
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
        }
    }

    public NN(NN nn)
    {
        sizes = new int[nn.sizes.Length];
        Array.Copy(nn.sizes, sizes, nn.sizes.Length);

        layers = new Layer[nn.layers.Length];
        //Array.Copy(nn.layers, layers, nn.layers.Length);
        for (int i = 0; i < layers.Length; i++)
            layers[i] = new Layer(nn.layers[i]);
    
        RMSpropCache = new Layer[layers.Length];
        for (int i = 1; i < sizes.Length; i++)
            RMSpropCache[i-1] = new Layer(sizes[i-1], sizes[i]);
        
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
    
    public double[][] FeedForward(double[][] inputs)
    {
        double[][] predicts = new double[inputs.GetLength(0)][];

        for (int i = 0; i < inputs.Length; i++)
            predicts[i] = FeedForward(inputs[i]);
        
        return predicts;
    }

    public double[] ExplicitOutput()
    {
        double[] outp = layers[layers.Length - 1].neurons.Apply(x => x < 0d ? -1d : 1d);

        return outp;
    }

    public void BackProp(double[][] inputs, double[][] errors)
    {
        Debug.Assert(inputs.Length == errors.Length);

        var batchSize = inputs.Length;

        var inputsMat = Matrix.ToMatrix(inputs);
        var errorsMat = Matrix.ToMatrix(errors);

        // Record whole NN neurons states for the every batch

        double[][,] neuronsStatesInLayers = new double[sizes.Length][,]; // sizes num x batch size x neurons in layer 

        for (int i = 0; i < sizes.Length; i++)
            neuronsStatesInLayers[i] = new double[batchSize, sizes[i]];
    
        neuronsStatesInLayers[0] = Matrix.Copy(inputsMat);
        
        for (int i = 0; i < batchSize; i++)
        {
            _ = FeedForward(inputs[i]);

            for (int j = 1; j < sizes.Length; j++)
                neuronsStatesInLayers[j] = neuronsStatesInLayers[j].SetRow(i, layers[j - 1].neurons);

        }

        // Calculating gradients 

        Layer[] gradBuff = new Layer[layers.Length];
        for (int i = 0; i < layers.Length; i++)
            gradBuff[i] = new Layer(layers[i].inputSize, layers[i].outSize);

        for (int k = layers.Length - 1; k >= 0; k--) // through layers
        {
            Layer l = layers[k];

            double[,] prevNeurons = neuronsStatesInLayers[k - 1 + 1];

            // Update weights

            gradBuff[k].weights = errorsMat.TransposeAndDot(prevNeurons).Transpose();
            gradBuff[k].biases = errorsMat.Sum(0);

            // Errors for the next layers

            var newErrs = new double[batchSize][];

            for (int i = 0; i < batchSize; i++)
                newErrs[i] = Matrix.Dot(l.weights.Transpose(), errors[i]);
            
            errorsMat = Matrix.ToMatrix(newErrs);
        }

        // Applying gradients with RMSprop optimization

        for (int i = 0; i < layers.Length; i++)
        {
            double[,] g = gradBuff[i].weights.Divide(batchSize);
            double[] gB = gradBuff[i].biases.Divide(batchSize);

            double[,] gSqr = g.Apply(x => x * x);
            double[] gSqrB = gB.Apply(x => x * x);
            
            RMSpropCache[i].weights = RMSpropCache[i].weights.Multiply(decayRate).Add(gSqr.Multiply(1d - decayRate));
            RMSpropCache[i].biases = RMSpropCache[i].biases.Multiply(decayRate).Add(gSqrB.Multiply(1d - decayRate));
            
            double[,] rmspropsqrt = RMSpropCache[i].weights.Apply(Math.Sqrt);
            double[] rmspropsqrtB = RMSpropCache[i].biases.Apply(Math.Sqrt);
            
            layers[i].weights = layers[i].weights.Add(g.Multiply(learningRate).Divide(rmspropsqrt.Add(1e-5)));
            layers[i].biases = layers[i].biases.Add(gB.Multiply(learningRate).Divide(rmspropsqrtB.Add(1e-5)));


            //layers[i].weights = layers[i].weights.Add(gradBuff[i].weights.Multiply(learningRate));
            //layers[i].biases = layers[i].biases.Add(gradBuff[i].biases.Multiply(learningRate));
        }

    }

    double arctanh(double x)
    {
        return 1f - (float)Math.Pow(Math.Tanh(x), 2f);
    }

}

