# face-tracking

Environment variables

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

| NAME                              | VALUE                                       |
|-----------------------------------|---------------------------------------------|
|shape_predictor_68_face_landmarks  |`\face-tracking\project`             		  |

DLIB instruction

1) Download https://github.com/davisking/dlib
2) Build the library and remember to set flags to ON in CMAKE:
	* DLIB_USE_CUDA
	* USE_SSE4_INSTRUCTIONS
	* USE_SSE2_INSTRUCTIONS
	* USE_AVX_INSTRUCTIONS