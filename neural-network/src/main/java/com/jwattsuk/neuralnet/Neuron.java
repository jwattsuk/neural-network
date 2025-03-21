package com.jwattsuk.neuralnet;

import java.util.Random;

class Neuron {
    Random random = new Random();


    public double compute(double input1, double input2){
        double preActivation = (this.weight1 * input1) + (this.weight2 * input2) + this.bias;
        double output = Util.sigmoid(preActivation);
        return output;
    }

    private Double oldBias = random.nextDouble(-1, 1), bias = random.nextDouble(-1, 1);
    public Double oldWeight1 = random.nextDouble(-1, 1), weight1 = random.nextDouble(-1, 1);
    private Double oldWeight2 = random.nextDouble(-1, 1), weight2 = random.nextDouble(-1, 1);

    public void mutate(){
        int propertyToChange = random.nextInt(0, 3);
        Double changeFactor = random.nextDouble(-1, 1);
        if (propertyToChange == 0){
            this.bias += changeFactor;
        } else if (propertyToChange == 1){
            this.weight1 += changeFactor;
        } else {
            this.weight2 += changeFactor;
        };
    }
    public void forget(){
        bias = oldBias;
        weight1 = oldWeight1;
        weight2 = oldWeight2;
    }
    public void remember(){
        oldBias = bias;
        oldWeight1 = weight1;
        oldWeight2 = weight2;
    }
}
