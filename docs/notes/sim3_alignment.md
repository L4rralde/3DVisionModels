## Sequence prediction

Consider a sequence of views (i.e., images of the same scene) $\{I_i\}$ with each image $I_i \in \mathbb R^{H \times W \times 3}$. Since memory is limited, we must partition the sequence into $n$ splits. By the moment, these splits overlap (to ease further alignment procedures). We'll denote these sequences of chunks by $\{\{I_i\}_0, \{I_i\}_2, \dots \}$. For each chunk, we obtain a prediction output using any of the 3D reconstruction foundation models. For readibility, we call it VGGT, but we can use MapAnything and DepthAnythingv3 as well.

$$
    \text{VGGT}(\{I_i\}_j) = \{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_j
$$

Where $D$ stands for depth maps, $D_{conf}$ belongs to the confidence maps of depth maps, $K$ corresponds to camera intrinsic matrices while $X = [R_i \text{ | } t_i]$ represents the extrinsics.

Actually, the set of predicted attributes may differ from model to model or even using different configurations. But this set of values suffice and are required (for further steps).

For instance, we can easily compute a ray representation using the camera intrinsics. On top of that, DepthAnythingv3 (DA3 for short) author's realized that computing the intrinsics and extrinsics using ray predictions attain better results but add overhead with iterative numerical optimization steps. Therefore, the default configuration uses a Camera Head to directly predict the Camera Intrinsics and Extrinsics (actually they predict: FOV parameters $f_i \in \mathbb R^2$, rotation as quaternion $q \in \mathbb R^4$, and a translation vector $t \in \mathbb R^3$); whilst VGGT and MapAnything include only the option of using a camera head to predict camera parameters (VGGT's camera head is way more complicated than DA3's since they designed a transformer which predicts a fixed-lenght sequence of estimations. Being the last estimation the best, and they compute the loss using the whole vector with a weighted loss function.)

### Point maps

Although we need point maps $P$ for 3D rendering, we don't need to either predict nor store them because it can be computed using the prediction outputs as follows:


$$
    P = R_i'(D_i(u, v) K_i^{-1} p) + t_i'
$$

Where $P=(x,y,z)^T$ is a 3D point in world coordinates and $p$ is a pixel in homogeneus pixel coordinate, i.e., $p =(u, v, 1)^T$. Note $D_i(u, v)$ is scalar.


Also note that **and $\mathbf{[R_i' \text{ | } t_i'] \neq X_i = [R_i \text{ | } t_i]}$**. This last inequality is of high importance because an extrinsic matrix is also the World-to-Camera Matrix transformation, which implies this is the transformation from the world coordinate system to the camera coordinate, in other words, the extrinsic matrices are the transformations to get the perspective of the scene from the camera point-of-view. It turns out $K_i^{-1} p$ are unit rays aligned to the camera (in the camera coordinate system). $D_i(u, v) K_i^{-1} p$ are the coordinates of each ray-depth end point from the perspective of the camera or the 3D point corresponding to each pixel in camera coordinates. We need the world coordinates, so we need to transform these points to world coordinates using the Cam-to-World transformation. The Cam-to-World transformation is the inverse of the World-to-Cam transformation or Extrinsics. Namely, 

$$
    [R_i' \text{ | } t_i'] = [R_i \text{ | } t_i]^{-1} = X_i^{-1} 
$$

Thus, we can also compute each point in world coordinates as follows:

$$
    P = X_i^{-1}
    \begin{bmatrix}
    D_i(u, v) K_i^{-1} p \\
    1
    \end{bmatrix}
$$


#### Depth-ray representation.

Predicting a valid rotation matrix is challenging due to the orthogonality constraint (rotation matrices are othonormal matrices). Results of higher quality are attained when using a per-pixel ray map $M \in \mathbb R^{H \times W \times 6}$ aligned to the input image and depth map. For each pixel $p$, a camera ray $r \in \mathbb R^6$ is defined by its origin $t_r\in \mathbb R^3$ and direction $d_r \in \mathbb R^3: r = (t_r, d_r)$. Note the direction $d_r$ is a unit vector.

The position of the 3D point from the camera perspective is the scaled depth-ray direction $D(u, v) d_r$. So, in the world coordinate, $P = t_r + D(u, v) d_r$. Finally, we reached the equality $d_r = R'K^{-1}p$, which means the direction $d_r$ is obtained by backprojecting $p$ into the camera frame and rotating it to the world frame.

We can compute the camera intrinsics and extrinsics (actually, we compute Cam-To-World transformation) from the dense ray map $M$. Since all rays must have a common origin, the origin of the camera  is simply the average of all rays' origin. But is quite more difficult to compute camera's orientation and intrinsics. In DA3 they solve an optimization problem and use QR decomposition.

**TO BE REVIEWED**

For a given view, we have an Intrinsic matrix $K$ and a rotation matrix $R_{cw}$ (Cam-to-World) ($R'$ in previous notation). Therefore,

$$
    d_r^* = (R_{cw}K^{-1})p = H^* p
$$

Both $d_r$ and $d_r^*$ are unit vectors. If $d_r$ from the dense ray map $M$ is a good estimation, then $d^* \times d \approx 0$ (they are parallel). For every pixel $p$ of an Image, we must find $H$ that minimizes:

$$
    H^* = \arg \min_{||H|| = 1} \sum_{h=1}^H \sum_{w=1}^W || Hp_{h, w} \times d_{r_{h, w}}||
$$

- Why $||H ||=1$?
- To maximize the dot product is not a better idea? (consider the sign)

Finally, $K$ (and $K^{-1}$ as well) is upper-triangular, $R_{cw}$ is orthonormal. We can get $K^{-1}$ and $R_{cw}$ with QR factorization ($H^* = R_{cw}K^{-1} = QR$).


## Chunks alignment

Again, consider the overlapping chunks predictions

$$
    \text{VGGT}(\{I_i\}_j) = \{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_j
$$

Say $\{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_A$ overlaps with $\{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_B$. We must find a transformation to align both chunks.

### $Sim3$ transformation.

$Sim3$ transformations are $S3$ transformations with scale change. The individual transformations are as follows

$$
S_{scale} = \begin{bmatrix}
        sI & 0 \\
        0 & 1
    \end{bmatrix} \\
S_{rot} = \begin{bmatrix}
        R & 0 \\
        0 & 1
    \end{bmatrix} \\
S_{trans} = \begin{bmatrix}
        I & t \\
        0 & 1
    \end{bmatrix} \\
$$

Akind to $S3$ transformations, we can concatenate them. For instance

$$
    S_{scale} S_{rigid} = \begin{bmatrix}
        sI & 0 \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        R & t \\
        0 & 1
    \end{bmatrix} =
    \begin{bmatrix}
        sR & st \\
        0 & 1
    \end{bmatrix}
$$

Nontheless, order still matters. For example, In VGGT-Long they use the following convention:

$$
    \begin{bmatrix}
        sR & t \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        p \\
        1
    \end{bmatrix}
    = 
    \begin{bmatrix}
        sRp +t \\
        1
    \end{bmatrix}
$$

Using composition, we have:

$$
    \begin{bmatrix}
        sR & t \\
        0 & 1
    \end{bmatrix}
    =
    \begin{bmatrix}
        I & t \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        s & 0 \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        R & 0 \\
        0 & 1
    \end{bmatrix}
    =
    S_{trans} S_{scale} S_{rot} = S_{trans} S_{rot} S_{trans}
$$

So the translation is in the final (destination) coordinates. Note that the rotation is invariant to scale transformation.

**Applying Sim3 transform for every chunk's attribute**


VGGT-Long's SIM3 transformagtion is used to align one chunk's point map to other chunk's point map, i.e., align one chunk's point map to the coordinate frame of the other. However, we want to track the whole prediction output, so we need to transform every attribute.

Consider we are moving and scaling the scene. When doing so, the attributes must be updated as listed below:

- Depth maps:
    Depth maps are 3D points in the cameras' coordinate system. So they are invariant to rigid transformation (moving the points) but not to scale change. We must scale each map:

    $$
        D \leftarrow s D
    $$

- $D_{conf}$:
    Confident matrices do not depend on the scale (since the predictions are not metric). So, these don't change at all.

    $$
        D_{conf} \leftarrow D_{conf}
    $$

- Intrinsics:
    Intrinsics are invariant to ~~both~~ rigid transformations ~~and scale changes~~. The former is due to the fact that intrinsics are not meant to include this information, but extrinsics. ~~And the latter happens because Intrinsics work with image coordinates~~.

- Extrinsics:
    Discuseed in section below.

- Point maps:
    Again, point maps don't need to be stored. They can be computed with the new depth maps, extrinsics and constant intrinsics. Nevertheless, we can use directly use Sim3 transformations.

    $$
        P \leftarrow S P
    $$


#### Aligning camera extrinsics.

Let $X_0$ and $X_1$ be extrinsics of the same view but from different chunks. We aim to find a procedure that aligns chunk 0 and chunk 1 such as $X_1' = X_0$. We do have the depth maps $D_0$ and $D_1$ so we may easily find $s^*$ such as $||D_0 - sD_1||_{Huber}$ is minimum.

