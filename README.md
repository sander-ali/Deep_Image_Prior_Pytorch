# Deep_Image_Prior_Pytorch
This repository provides the code for training deep image prior networks for image denoising with pytorch.  
Deep Image Prior Networks works on few-shot and one-shot denoising principles, i.e. it does not require datasets to train for separating noise from images.  
The implementation is based on the paper (Deep Image Prior)[https://arxiv.org/abs/1711.10925] by Ulyanov et al.  
The network first reconstructs the target image from random intialized noisy image. The network then computes the loss between reconstructed and original noisy image to update the network. Post some iterations, the network will start enhancing/denoising the image in contrast to the original one. The phenomenon is observed as the network is optimized to withhold the prior information from the image, hence the name, deep image prior. Iteration wise results from the code are shown below.  

Iteration # 100  
![iter100](https://user-images.githubusercontent.com/26203136/186658812-d260dc76-a735-412d-867e-7d07120cd868.png)  
Iteration # 500  
![iter500](https://user-images.githubusercontent.com/26203136/186658845-f3d296b0-cfc0-4a43-b47a-f0515f9f46ad.png)  
Iteration # 1000  
![iter1000](https://user-images.githubusercontent.com/26203136/186658970-ac375242-fc3e-47d0-ad51-fcfac90f1f9c.png)  
Iteration # 1500  
![iter1500](https://user-images.githubusercontent.com/26203136/186659023-ea172aa4-13af-4d71-8755-73f07743d9b2.png)  
Iteration # 2000  
![iter2000](https://user-images.githubusercontent.com/26203136/186659097-9d7bc8c7-1f5d-44ac-aea5-7bf757466db4.png)  
Iteration # 2500  
![iter2500](https://user-images.githubusercontent.com/26203136/186659146-094c84b7-2087-4f2a-aa56-a252b1b4bdec.png)  
Iteration # 3000  
![iter3000](https://user-images.githubusercontent.com/26203136/186659205-e5c6ff66-4766-45e8-be2a-1499571c44d0.png)  
Iteration # 3500  
![iter3500](https://user-images.githubusercontent.com/26203136/186659242-95df43f5-5171-4420-8a22-c945dfbeb97c.png)  
Iteration # 4000  
![iter4000](https://user-images.githubusercontent.com/26203136/186659293-b2c6039a-edac-4852-b76b-ef8252d50393.png)  
Iteration # 4500  
![iter4500](https://user-images.githubusercontent.com/26203136/186659361-c1bf648d-7a0f-4096-a334-73e5616ac107.png)  
Iteration # 5000  
![iter5000](https://user-images.githubusercontent.com/26203136/186659409-cd75d05b-8868-4c24-bd5b-832f7ca5d45e.png)
