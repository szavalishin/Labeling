Labeling Demo
Copyright (c) Sergey Zavalishin 2010-2015

	This is a demo application for “Block equivalence algorithm for labeling 
	2D and 3D images on GPU” paper, which demonstrates GPU-based block equivalence 
	labeling algortihm for 2D and 3D cases. It also contains several other 
	algortihms, which were implemented during development of BE algorithm.


The demo implements the following algorithms:

	[1] S. Zavalishin, I. Safonov, Y. Bekhtin, I. Kurilin, “Block equivalence 
		algorithm for labeling 2D and 3D images on GPU”, Electronic Imaging 2016, 
		Visual Information Processing and Communication Conference
	[2] Grana, Costantino, Daniele Borghesani, and Rita Cucchiara. "Optimized 
		block-based connected components labeling with decision trees." Image 
		Processing, IEEE Transactions on 19.6 (2010): 1596-1609.
	[3] He, Lifeng, Yuyan Chao, and Kenji Suzuki. "A run-based two-scan labeling 
		algorithm." Image Processing, IEEE Transactions on 17.5 (2008): 749-756.
	[4] Y. Bekhtin, V. Gurov, S. Zavalishin, “A Run Equivalence Algorithm for 
		Parallel Connected Component Labeling on CPU”, Embedded Computing (MECO), 
		2015 4th Mediterranean Conference, pp.276-279
	[5] Kalentev, Oleksandr, et al. "Connected component labeling on a 2D grid 
		using CUDA." Journal of Parallel and Distributed Computing 71.4 (2011): 
		615-620.
	[6] Wu, Kesheng, Ekow Otoo, and Kenji Suzuki. "Two strategies to speed up 
		connected component labeling algorithms." Lawrence Berkeley National 
		Laboratory (2008).

Notes:

	[2] is based on the code from http://phaisarn.com/labeling
	[3] is a relatively slow version, which was implemented for the reference only
	[6] is taken from OpenCV 3.0.0 implementation, thus this algorithm may be
	    changed in further versions of OpenCV


Compilation:

	The demo requires the following third party components:

		* OpenCV 3.0.0
		* boost 1.55.0
		* OpenCL 1.1

	Originally it was designed for OpenCL headers from CUDA 7.0 , but there's
	no specific restrictions on using any other compatible headers from other
	OpenCV vendors.

Usage:

	Detailed help is embedded into the demo. Use '-h' flag to get it.
	The basic scenario includes labeling for the single image, which may be
	performed using the following flag combination:

		labeling -a bleq -g -i "image.png" -o "out_path"

	It means that image.png will be labeled using GPU version of Block 
	Equivalence algorithm [1] and the result stored at "out_path". Note, 
	that storing labeled image is a time consuming process, thus try to 
	avoid using it.

	Another scenario includes batch processing:

		labeling -a gr-block -i "in_path"

	Using these flags you can label all images from "in_path" with Grana
	alrotihm [2].

	Finally, you may label 3D images stored as a set of 2D slices:

		labeling -a lbeq -3 -g "in_image"

	These flags mean that input image from "in_image" directory will be
	labeled using Label Equivalence algortihm [5] on GPU.
