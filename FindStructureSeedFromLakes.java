package neil;

import neil.dungeons.kaptainwutax.magic.PopReversal2TheHalvening;
import neil.dungeons.kaptainwutax.magic.RandomSeed;
import neil.dungeons.kaptainwutax.util.LCG;
import neil.dungeons.kaptainwutax.util.Rand;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/*
** Last stage of approach to find 'Siege on Castle Steve Video' World Seed.
** Written by pablion (pabli0n)
** Date: 2024-01-03
*/

public class FindStructureSeedFromLakes {

	public static class Data {
		static final LCG nextSeed = Rand.JAVA_LCG.combine(-1);
		public long dungeonSeed;
		public int posX;
		public int posZ;

		public Data(long dungeonSeed, int posX, int posZ) {
			this.dungeonSeed = dungeonSeed;
			this.posX = posX;
			this.posZ = posZ;
		}

		public long getPrevious() {
			return dungeonSeed = nextSeed.nextSeed(dungeonSeed);
		}
	}


	public static void main(String[] args) {
		
		List<Integer> xV = new ArrayList<>();
		List<Integer> zV = new ArrayList<>();
		
		int lastChCoord = 0;
		
		for(int X = -5000; X <= 5000; X++)
		{
			if(lastChCoord != (X >> 4))
			{
				lastChCoord = X >> 4;
		    
		    	xV.add(lastChCoord);
			}
		}
		
		System.out.println();
		
		lastChCoord = 0;
		
		for(int X = -5000; X <= 5000; X++)
		{
			if(lastChCoord != (X  >> 4))
			{
				lastChCoord = X >> 4;
		
				zV.add(lastChCoord);
			}
		}
		
		IntStream.range(0, xV.size() * zV.size()).parallel().forEach((index) ->//for(int x : xV)
		{
			int x = index % xV.size();
			
			int z = index / xV.size();
			
			//IntStream.range(0, zV.size()).parallel().forEach((z) ->//for(int z : zV)
			{
		 		List<Data> dataList = new ArrayList<>();
		     	 
		 		dataList.add(new Data(26986154423438l, xV.get(x) * 16, zV.get(z) * 16));

				List<Long> res = crack(dataList);
				
				System.out.println("Done for " + (xV.get(x) * 16) + ":" + (zV.get(z) * 16));
				
				System.out.println("Iteration: " + iteration + "/" + (xV.size() * zV.size()));
				
				iteration++;

				for (long seed : res) {
					
					long worldSeed = (0 << 48) | seed;
					
					Random random = new Random();
					
					random.setSeed(worldSeed);
					
			        long l1 = (random.nextLong() / 2L) * 2L + 1L;
			        long l2 = (random.nextLong() / 2L) * 2L + 1L;
			        random.setSeed((long)xV.get(x) * l1 + (long)zV.get(z) * l2 ^ worldSeed);
			        
		            if(random.nextInt(4) == 0) // could be only that line...
		            {
		            	random.nextInt(16); random.nextInt(128); random.nextInt(16);

		                int l = random.nextInt(4) + 4;

		                for(int i1 = 0; i1 < l; i1++)
		                {
		                    double d = random.nextDouble() * 6 + 3;
		                    double d1 = random.nextDouble() * 4 + 2;
		                    double d2 = random.nextDouble() * 6 + 3;
		                    double d3 = random.nextDouble() * (16 - d - 2) + 1.0 + d * 0.5;
		                    double d4 = random.nextDouble() * (8 - d1 - 4) + 2 + d1 * 0.5;
		                    double d5 = random.nextDouble() * (16 - d2 - 2) + 1.0 + d2 * 0.5;
		                }
		            }
		            
		            if(random.nextInt(8) == 0)
		            {
		            	if(getSeed(random) == 26986154423438l)
		            	{
		        			File file2 = new File("<<dir>>/worldseeds.txt");
		        			
		        			if(!file2.exists())
		        			{
		        				try
		        				{
		        					file2.createNewFile();
		        				}
		        				catch(IOException exception)
		        				{
		        					exception.printStackTrace();
		        				}
		        			}
		        			
		        			try
		        			{
		        				PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(file2, true)));
		        				
		        				out.println(worldSeed + "/" + xV.get(x) + "/" + zV.get(z));
		        				
		        				out.close();
		        			}
		        			catch(Exception exception)
		        			{
		        				exception.printStackTrace();
		        			}
		            	}
		            }
				}
			//});
			}
		});
	     
	     
	    System.out.println("Done all!");
	    
	    System.exit(0);
	}
	
	public static List<Long> crack(List<Data> dataList) {
		List<Long> res = new ArrayList<>();
		Map<Long, Boolean> map = new HashMap<>();
		
		for(int i = 0; i < 2; i++)
		{
			for (Data data : dataList) {
				PopReversal2TheHalvening.getSeedFromChunkseedPre13(
				data.getPrevious() ^ Rand.JAVA_LCG.multiplier,
				data.posX >> 4, data.posZ >> 4).forEach(e -> {
					//System.out.println("Found structure seed: " + e);
					res.add(e);
					
					map.remove(e);
				});
			}
		}
		return res;
	}
	
	public static void pushSeed(long seed, Random rand)
	{
        try
        {
            Field field = Random.class.getDeclaredField("seed");
            field.setAccessible(true);
            
            field.set(rand, new AtomicLong(seed));
        }
        catch (Exception e)
        {
            //handle exception
        }
	}
	
	public static long getSeed(Random rand)
	{
		AtomicLong theSeed = null;
        try
        {
            Field field = Random.class.getDeclaredField("seed");
            field.setAccessible(true);
            
            theSeed = (AtomicLong)field.get(rand);
        }
        catch (Exception e)
        {
            //handle exception
        }
        
        return theSeed.get();
	}

	public static void skip(int n, Random rand)
	{
		long multiplier = 1;
		long addend = 0;

		long intermediateMultiplier = 25214903917L;
		long intermediateAddend = 11;

		for (long k = n; k != 0; k >>>= 1) {
			if ((k & 1) != 0) {
				multiplier *= intermediateMultiplier;
				addend = intermediateMultiplier * addend + intermediateAddend;
			}

			intermediateAddend = (intermediateMultiplier + 1) * intermediateAddend;
			intermediateMultiplier *= intermediateMultiplier;
		}
		

		multiplier &= (1L << 48) - 1;
		addend &= (1L << 48) - 1;
		
        long theSeed;
        try
        {
            Field field = Random.class.getDeclaredField("seed");
            field.setAccessible(true);
            AtomicLong scrambledSeed = (AtomicLong) field.get(rand);
            theSeed = scrambledSeed.get();
            field.set(rand, new AtomicLong((theSeed * multiplier + addend) & ((1L << 48) - 1)));
        }
        catch (Exception e)
        {
            //handle exception
        }
	}
	
	public static int iteration;
}

