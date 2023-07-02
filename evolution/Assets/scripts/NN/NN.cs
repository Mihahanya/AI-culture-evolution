using Accord.Math;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class NN
{
    public Layer[] layers;
    private LayerData[] RMSpropCache;
    public int[] sizes;
    public double learningRate = 0.003f;
    public double decayRate = 0.99;

    public NN(params int[] sizes)
    {
        this.sizes = sizes;

        layers = new Layer[sizes.Length-1];
        RMSpropCache = new LayerData[layers.Length];
        for (int i = 1; i < sizes.Length; i++)
        {
            layers[i-1] = new Layer(sizes[i-1], sizes[i]);
            layers[i-1].initRandom(-1f, 1f);
            
            RMSpropCache[i-1] = new LayerData(sizes[i-1], sizes[i]);
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
    
        RMSpropCache = new LayerData[layers.Length];
        for (int i = 1; i < sizes.Length; i++)
            RMSpropCache[i-1] = new LayerData(sizes[i-1], sizes[i]);
        
    }

    public double[] FeedForward(double[] inputs)
    {
        Debug.Assert(inputs.Length == sizes[0]);

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

    public double[][] GetState() // without inputs
    {
        double[][] state = new double[layers.Length][];

        for (int i = 0; i < layers.Length; i++)
        {
            state[i] = new double[layers[i].neurons.Length];
            Array.Copy(layers[i].neurons, state[i], layers[i].neurons.Length);
        }

        return state;
    }

    private LayerData[] CalculateGradients(double[][,] nnStates, double[,] errors, int batchSize)
    {
        var errorsMat = errors.Clone() as double[,];

        LayerData[] grads = new LayerData[layers.Length];
        for (int i = 0; i < layers.Length; i++)
            grads[i] = new LayerData(layers[i].inputSize, layers[i].outSize);

        for (int k = layers.Length - 1; k >= 0; k--) // through layers
        {
            Layer l = layers[k];

            double[,] prevNeurons = nnStates[k - 1 + 1];

            // Update weights

            grads[k].weights = errorsMat.TransposeAndDot(prevNeurons).Transpose();
            grads[k].biases = errorsMat.Sum(0);

            // Errors for the next layers

            if (k != 0)
            {
                var newErrs = new double[batchSize][];

                for (int i = 0; i < batchSize; i++)
                    newErrs[i] = Matrix.Dot(l.weights.Transpose(), errorsMat.GetRow(i));

                errorsMat = Matrix.ToMatrix(newErrs);
            }
        }

        return grads;
    }

    public void BackProp(double[][,] states, double[,] errors) // sizes num x batch size x neurons in layer
    {
        var batchSize = errors.GetLength(0);

        LayerData[] grads = CalculateGradients(states, errors, batchSize);

        ApplyGradientsRMSprop(grads, batchSize);
    }
    
    public void BackProp(double[][][] states, double[,] errors) // sizes num x batch size x neurons in layer
    {
        double[][,] formStates = new double[states.Length][,];
        for (int i = 0; i < states.Length; i++)
        {
            formStates[i] = states[i].ToMatrix().Clone() as double[,];
        }

        BackProp(formStates, errors);
    }

    public void BackProp(double[][] inputs, double[,] errors)
    {
        Debug.Assert(inputs.Length == errors.GetLength(0));
        Debug.Assert(inputs[0].Length == sizes[0]);

        var batchSize = inputs.Length;

        // Record whole NN neurons states for the every batch

        double[][,] neuronsStatesInLayers = new double[sizes.Length][,]; // sizes num x batch size x neurons in layer 

        for (int i = 0; i < sizes.Length; i++)
            neuronsStatesInLayers[i] = new double[batchSize, sizes[i]];

        for (int i = 0; i < batchSize; i++)
        {
            _ = FeedForward(inputs[i]);

            for (int j = 1; j < sizes.Length; j++)
                neuronsStatesInLayers[j] = neuronsStatesInLayers[j].SetRow(i, layers[j - 1].neurons);
        }
    
        neuronsStatesInLayers[0] = Matrix.ToMatrix(inputs).Clone() as double[,];

        BackProp(neuronsStatesInLayers, errors);
    }

    private void ApplyGradientsRMSprop(LayerData[] grads, int batchSize=1)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            double[,] g = grads[i].weights;
            double[] gB = grads[i].biases;

            double[,] gSqr = g.Pow(2);
            double[] gSqrB = gB.Apply(x => x * x);

            RMSpropCache[i].weights = RMSpropCache[i].weights.Multiply(decayRate).Add(gSqr.Multiply(1d - decayRate));
            RMSpropCache[i].biases = RMSpropCache[i].biases.Multiply(decayRate).Add(gSqrB.Multiply(1d - decayRate));

            double[,] rmspropsqrt = RMSpropCache[i].weights.Apply(Math.Sqrt);
            double[] rmspropsqrtB = RMSpropCache[i].biases.Apply(Math.Sqrt);

            layers[i].weights = layers[i].weights.Add(g.Multiply(learningRate / batchSize).Divide(rmspropsqrt.Add(1e-5)));
            layers[i].biases = layers[i].biases.Add(gB.Multiply(learningRate / batchSize).Divide(rmspropsqrtB.Add(1e-5)));
        }
    }

    private void ApplyGradientsSGD(LayerData[] grads, int batchSize=1)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].weights = layers[i].weights.Add(grads[i].weights.Multiply(learningRate / batchSize));
            layers[i].biases = layers[i].biases.Add(grads[i].biases.Multiply(learningRate / batchSize));
        }
    }

    double arctanh(double x)
    {
        return 1f - (float)Math.Pow(Math.Tanh(x), 2f);
    }

}

