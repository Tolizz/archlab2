
// define the T1, T2 threshold constants
#define T1 32
#define T2 96

// table dimensions
#define HEIGHT 128
#define WIDTH 128

#define IMAGE_SIZE (HEIGHT * WIDTH)

int clip (int x, int min_val, int max_val){

#pragma HLS INLINE

	if(x < min_val){
		return min_val;
	}
	else if (x > max_val){
		return max_val;
	}
	else{
		return x;
	}
}

extern "C" {
void IMAGE_DIFF_POSTERIZE(int *in1, // Input Image A
          	  	  	  	  int *in2, // Input Image B
						  int *out_r     // Output Image C_filter
          	  	  	  	  ) {
// Here Vitis kernel contains one s_axilite interface which will be used by host
// application to configure the kernel.
// Here bundle control is defined which is s_axilite interface and associated
// with all the arguments (in1, in2, out_r and size),
// control interface must also be associated with "return".
// All the global memory access arguments must be associated to one m_axi(AXI
// Master Interface). Here all three arguments(in1, in2, out_r) are
// associated to bundle gmem which means that a AXI master interface named
// "gmem" will be created in Kernel and all these variables will be
// accessing global memory through this interface.
// Multiple interfaces can also be created based on the requirements. For
// example when multiple memory accessing arguments need access to
// global memory simultaneously, user can create multiple master interfaces and
// can connect to different arguments.
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out_r offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out_r bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control


	//assign local BRAM
	int img_A[HEIGHT][WIDTH];	// image A
	int img_B[HEIGHT][WIDTH];	// image B
	int img_C[HEIGHT][WIDTH]; 	// posterized image
	int C_filt[HEIGHT][WIDTH];	// filtered image

#pragma HLS ARRAY_PARTITION variable=img_A cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=img_B cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=img_C cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=C_filt cyclic factor=16 dim=2


  read_A:
  for(int i = 0; i < HEIGHT; i++){
	  for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE II = 1
		  img_A[i][j] = in1[i * WIDTH + j];
	  }
  }


  read_B:
  for(int i = 0; i < HEIGHT; i++){
	  for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE II = 1
		  img_B[i][j] = in2[i * WIDTH + j];
	  }
  }

    //image posterization
    process_diff_post:
	for(int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor=16

			int val_a = img_A[i][j];
			int val_b = img_B[i][j];
			int diff = val_a - val_b;
			int abs_diff = (diff > 0) ? diff : -diff;

			if (abs_diff < T1)
				img_C[i][j] = 0;
			else if (abs_diff < T2)
				img_C[i][j] = 128;
			else
				img_C[i][j] = 255;
    	}
	}

	// applying the sharpening filter
	process_sharpen_filter:
	for(int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE II=1
#pragma UNROLL factor=16
			if(i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1){
				C_filt[i][j] = img_C[i][j];
			}
			else {
				int center = img_C[i][j];
				int up = img_C[i-1][j];
				int down = img_C[i+1][j];
				int left = img_C[i][j-1];
				int right = img_C[i][j+1];

				int val = (5 * center) - up - down - left - right;
				C_filt[i][j] = clip(val, 0, 255);
			}
		}
	}



	//write the result
	for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++){
#pragma HLS PIPELINE II=1
			out_r[i * WIDTH + j] = C_filt[i][j];
		}
	}
}
}
