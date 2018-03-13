import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;

import shared.Instance;

public class DataParser {
	
	private DataParser(){	
	}
	
	public static Instance[] getData(String path, int data_length, int attributes_length) {

        double[][][] attributes = new double[data_length][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(path)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[attributes_length]; // attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < attributes_length; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
	}
}
