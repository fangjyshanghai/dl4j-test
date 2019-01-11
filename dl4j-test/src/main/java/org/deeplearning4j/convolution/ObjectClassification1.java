package org.deeplearning4j.convolution;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.channels.Channels;
import java.util.Random;

/**
 * Using Convolutional Neural Network for classification of three classes [dog, donut, earth]
 *
 * Search for the part with *Enter your code here* and replace with model configuration
 *
 * [NOTE: Do not change other parts other than function getConfig(...)]
 */
public class ObjectClassification
{
    private static final Logger log = LoggerFactory.getLogger(ObjectClassification.class);

    protected static int height = 100;              //Image height
    protected static int width = 100;               //Image width
    protected static int channels = 3;              //Image depth
    protected static int classes = 3;               //Number of classes
    protected static int batchSize = 25;

    protected static long seed = 42;                //Seed number for reproduction
    protected static Random rng = new Random(seed);

    protected static double trainDataRatio = 0.8;   //Segregate data into training and testing dataset

    protected static int epochs = 20;               //Number of epochs

    public void run() throws Exception
    {
        //Load model if exist. Test on single image
        String rootPath = new ClassPathResource("/classification").getFile().toString();
        File saveAs = new File(rootPath + "/objectModel.zip");


        //Set image root directory
        File rootDir = new File(rootPath + "/threeobjects");
        FileSplit fileSplit = new FileSplit(rootDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        if(rootDir.exists() == false)
        {

            System.out.println("File not exist. Abort");
            return;
        }

        //Get images directory' name as label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //Get number of labels by number of directory in the rootDir. The rootDir must not contain other contents.
        System.out.println(rootDir.toString());
        int numLabels = rootDir.listFiles(File::isDirectory).length;

        //Split into training and testing file split, images of different labels shuffled here
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS, labelMaker);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, trainDataRatio, 1 - trainDataRatio);
        InputSplit trainSplit = inputSplit[0];
        InputSplit testSplit = inputSplit[1];

        //Set image record reader for training and testing data
        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMaker);
        rrTrain.initialize(trainSplit);

        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMaker);
        rrTest.initialize(testSplit);

        //Set data iterator
        DataSetIterator iterTrain = new RecordReaderDataSetIterator(rrTrain, batchSize, 1, numLabels);
        DataSetIterator iterTest = new RecordReaderDataSetIterator(rrTest, batchSize, 1, numLabels);


        //Data normalization
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterTrain);
        iterTrain.setPreProcessor(scaler);
        iterTest.setPreProcessor(scaler);


//        MultiLayerConfiguration config = getConfig(channels, classes);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()

/*                .seed(seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                //.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(200)
                        .nOut(20)
                        //.stride(1, 10)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                //.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden laye
                        .nIn(20)
                        .nOut(8)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        //.setInputType(InputType.convolutional(height,width, Channels))
                        .build())
                 .setInputType(InputType.convolutionalFlat(3, 3, 1))
                .backprop(true).pretrain(false)
  */

       // MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.iterations(iterations)
                //.regularization(true).l2(0.0005)
                //.learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(4, 1)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        //.name("hzvt1")
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(26)
                        .activation(Activation.RELU) //.activation("identity")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(classes)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height,width,channels))
                .backprop(true).pretrain(false)
                .build();

/**
        if(config == null)
        {
            System.out.println("Configuration not set right. Abort");
            return;
        }
*/
        //Build model
        log.info("Build model....");

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        network.setListeners(new ScoreIterationListener(10));

        //Start training
        log.info("Train model....");

        for (int i = 0; i < epochs; ++i)
        {
            network.fit(iterTrain);
        }

        //Evaluate model
        Evaluation eval = network.evaluate(iterTest);
        log.info(eval.stats());

        //Save model
        ModelSerializer.writeModel(network, saveAs, false);

        log.info("Program end.");
    }


    /**
     * Build network configuration
     *
     * @param numInputs  input layer nodes
     * @param numOutputs output layer nodes
     * @return MultiLayerConfiguration with network configuration
     */
  //  public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs) {

        /**
         * Enter your code here
         */

/**        ''
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numInputs)
                        .nOut(80)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(80)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        return null; //change to return MultiLayerConfiguration instance
    }
*/

    public static void main(String[] args) throws Exception
    {
        ObjectClassification classifier = new ObjectClassification();

        classifier.run();
    }

}
