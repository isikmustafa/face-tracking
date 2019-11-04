# face-tracking

Environment variables

Side notes:
Use `OpenCV 4.0.0`

| NAME            | EXAMPLE VALUE                                                 |
|-----------------|---------------------------------------------------------------|
|GLFW3            |`\face-tracking\3rd\glfw-3.3\bin\lib-vc2017-64bit`             |
|OPENCV_DIR       |`\OpenCV\opencv\build\x64\vc15`                                |
|OPENCV_INCLUDE   |`\OpenCV\opencv\build\include`                                 |
|OPENCV_BIN 	  |`\OpenCV\opencv\build\x64\vc15\bin`                            |
|OPENCV_LIB 	  |`\OpenCV\opencv\build\x64\vc15\lib`                            |
|DLIB_DIR         |`\Projects\dlib`                                               |
|DLIB_LIB_DEBUG	  |`\Projects\dlib\examples\build\dlib_build\Debug`               |
|DLIB_LIB_RELEASE |`\Projects\dlib\examples\build\dlib_build\Release`             |


Extra files

| NAME                                  | VALUE                                    |
|---------------------------------------|------------------------------------------|
|shape_predictor_68_face_landmarks.dat  |`\face-tracking\project`             	   |

DLIB instruction

1) Download https://github.com/davisking/dlib Commit hash: e30f5e2fe88ff3a9c8265c6265b37956ce535ac2
2) Build examples and remember to set flags to ON in CMAKE:
	* DLIB_USE_CUDA
	* USE_SSE4_INSTRUCTIONS
	* USE_SSE2_INSTRUCTIONS
	* USE_AVX_INSTRUCTIONS
3) Build one of the project example in VS to generate dlib_build folder

References
* [A Morphable Model For The Synthesis Of 3D Faces](https://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf)
* [Real-time Expression Transfer for Facial Reenactment](http://zollhoefer.com/papers/SGA2015_Face/paper.pdf)
* [Face2Face: Real-time Face Capture and Reenactment of RGB Videos](https://web.stanford.edu/~zollhoef/papers/CVPR2016_Face2Face/paper.pdf)