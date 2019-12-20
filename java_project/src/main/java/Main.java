import java.awt.Canvas;
import java.awt.Graphics;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

import javax.swing.JFrame;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main extends Canvas {

	public JFrame frame;

	public Main() {
		frame = new JFrame("Realtime");
		frame.setSize(187, 200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(this);
		frame.setVisible(true);
	}

	static ArrayList<double[]> x = new ArrayList<double[]>();
	static ArrayList<Double> y = new ArrayList<Double>();

	static int currX = 0;

	public static void main(String[] args) throws Exception {
		Main main = new Main();

		BufferedReader reader = new BufferedReader(new FileReader("mitbih_test.csv"));
		String line = reader.readLine();
		while (line != null) {
			String[] input = line.split(",");
			double[] thisX = new double[187];
			for (int i = 0; i < 187; i++) {
				thisX[i] = Double.parseDouble(input[i]);
			}
			x.add(thisX);
			y.add(Double.parseDouble(input[187]));
			line = reader.readLine();
		}

		// main.repaint();

		MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("model.json", "model.h5");
		double sum = 0;
		double count = 0;
		long[] times = new long[x.size()];
		for (int i = 0; i < x.size(); i++) {
			main.repaint();
			currX++;
			long last_time = System.currentTimeMillis();
			INDArray arr1 = Nd4j.create(new double[][][] { { x.get(i) } });
			INDArray output = model.output(arr1);
			times[i] = System.currentTimeMillis() - last_time;
			sum += times[i];
			count++;
		}
		System.out.println("Mean time (ms): " + (sum / count));
		System.out.println("SD of times (ms): " + calculateSD(times));
	}

	public static double calculateSD(long numArray[]) {
		double sum = 0.0, standardDeviation = 0.0;
		int length = numArray.length;
		for (double num : numArray) {
			sum += num;
		}
		double mean = sum / length;
		for (double num : numArray) {
			standardDeviation += Math.pow(num - mean, 2);
		}
		return Math.sqrt(standardDeviation / length);
	}

	@Override
	public void paint(Graphics g) {
		super.paint(g);
		for (int i = 0; i < 186; i++) {
			g.drawLine(i, 200 - (int) (x.get(currX)[i] * 200.0), i + 1, 200 - (int) (x.get(currX)[i + 1] * 200.0));
		}
	}

}
