## Sequence prediction

Consider a sequence of views (i.e., images of the same scene) $\{I_i\}$ where each image $I_i \in \mathbb R^{H \times W \times 3}$. Since memory is limited, we must partition the sequence into $n$ splits. By the moment, these splits overlap (to ease further alignment procedures). We'll denote these sequences of chunks by $\{\{I_i\}_0, \{I_i\}_2, \dots \}$. For each chunk, we obtain a prediction output using any of the 3D reconstruction foundation models. For readibility, we call it VGGT, but we can use MapAnything and DepthAnythingv3 as well.

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


Also note that **and $\mathbf{[R_i' \text{ | } t_i'] \neq X_i = [R_i \text{ | } t_i]}$**. This last inequality is of high importance because an extrinsic matrix is also the World-to-Camera Matrix transformation, which implies this is the transformation from the world coordinate system to the camera coordinate, in other words, the extrinsic matrices are the transformations to get the perspective of the scene from the camera point-of-view. It turns out $K_i^{-1} p$ are ~~unit~~ rays aligned to the camera (in the camera coordinate system). $D_i(u, v) K_i^{-1} p$ are the coordinates of each ray-depth end point from the perspective of the camera or the 3D point corresponding to each pixel in camera coordinates. We need the world coordinates, so we need to transform these points to world coordinates using the Cam-to-World transformation. The Cam-to-World transformation is the inverse of the World-to-Cam transformation or Extrinsics. Namely, 

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

Predicting a valid rotation matrix is challenging due to the orthogonality constraint (rotation matrices are othonormal matrices). Results of higher quality are attained when using a per-pixel ray map $M \in \mathbb R^{H \times W \times 6}$ aligned to the input image and depth map. For each pixel $p$, a camera ray $r \in \mathbb R^6$ is defined by its origin $t_r\in \mathbb R^3$ and direction $d_r \in \mathbb R^3: r = (t_r, d_r)$.

The position of the 3D point from the camera perspective is the scaled depth-ray direction $D(u, v) d_r$. So, in the world coordinate, $P = t_r + D(u, v) d_r$. Finally, we reached the equality $d_r = R'K^{-1}p$, which means the direction $d_r$ is obtained by backprojecting $p$ into the camera frame and rotating it to the world frame.

We can compute the camera intrinsics and extrinsics (actually, we compute Cam-To-World transformation) from the dense ray map $M$. Since all rays must have a common origin, the origin of the camera  is simply the average of all rays' origin. But is quite more difficult to compute camera's orientation and intrinsics. In DA3 they solve an optimization problem and use QR decomposition.

**TO BE REVIEWED**

For a given view, we have an Intrinsic matrix $K$ and a rotation matrix $R_{cw}$ (Cam-to-World) ($R'$ in previous notation). Therefore,

$$
    d_r^* = (R_{cw}K^{-1})p = H^* p
$$

~~Both $d_r$ and $d_r^*$ are unit vectors~~. If $d_r$ from the dense ray map $M$ is a good estimation, then $d^* \times d \approx 0$ (they are parallel). For every pixel $p$ of an Image, we must find $H$ that minimizes:

$$
    H^* = \arg \min_{||H|| = 1} \sum_{h=1}^H \sum_{w=1}^W || Hp_{h, w} \times d_{r_{h, w}}||
$$

- Why $||H ||=1$?
- To maximize the dot product is not a better idea? (consider the sign)

Finally, $K$ (and $K^{-1}$ as well) is upper-triangular, $R_{cw}$ is orthonormal. We can get $K^{-1}$ and $R_{cw}$ via QR factorization ($H^* = R_{cw}K^{-1} = QR$).


## Chunks alignment

Again, consider the overlapping chunks predictions

$$
    \text{VGGT}(\{I_i\}_j) = \{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_j
$$

Say $\{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_A$ overlaps with $\{\{D_i\}, \{D_{conf_i}\}, \{K_i\}, \{X_i\}\}_B$. We must find a transformation to align both chunks.

### $Sim(3)$ transformation.

$Sim(3)$ transformations are $SE(3)$ transformations with scale change. The individual transformations are as follows

$$
S_{s} = \begin{bmatrix}
        sI & 0 \\
        0 & 1
    \end{bmatrix} \\
S_{R} = \begin{bmatrix}
        R & 0 \\
        0 & 1
    \end{bmatrix} \\
S_{t} = \begin{bmatrix}
        I & t \\
        0 & 1
    \end{bmatrix} \\
$$


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

The following operations may be frequently used:

$$
    S_{rigid} = T = \begin{bmatrix}
        R & t \\
        0 & 1
    \end{bmatrix}
$$

As we normally describe any possible rigid transformation as above, we also denote any possible $Sim(3)$ transformation $S$ as follows:

$$
    S = 
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
    S_{t} S_{s} S_{R} = S_{t} S_{R} S_{t}
$$

So the translation is in the final (destination) coordinates. Note that the rotation is invariant to scale transformation.

**Applying Sim(3) transform for every chunk's attribute**


VGGT-Long's Sim(3) transformation is used to align one chunk's point map to other chunk's point map, i.e., align one chunk's point map to the coordinate frame of the other. However, we want to track the whole prediction output, so we need to transform every attribute.

First, let's understand how the predictions ressembles the conventional, and withou aberrations, image projection procedure.

A point in the world's coordinate system $P_W = \begin{bmatrix}X_W & Y_W & Z_W \end{bmatrix}^T$ is projected into the pixel at position $u, v$ as follows:

$$
    \begin{bmatrix}
        u \\ v \\ 1
    \end{bmatrix}
    =
    \frac{1}{Z_c} K
    \begin{bmatrix} 
        R & t
    \end{bmatrix}
    \begin{bmatrix}
        X_W \\ Y_W \\ Z_w
    \end{bmatrix}
$$

Being $K$ the intrinsics matrix.

The extrinsics $X$ is:
$$
    X = \begin{bmatrix}
        R & t \\
        0 & 1
    \end{bmatrix}
$$

And the suffix $_C$ stands for the coordinates from the camera $C$ point of view. Which follows:

$$
    \begin{bmatrix}
        P_C \\ 1
    \end{bmatrix}
    =
    X 
    \begin{bmatrix}
        P_W\\1
    \end{bmatrix}
$$

In words, $K$ projects a point from the camera perspective into its image in pixel units. $X$ maps a point in the world's coordinate system into the camera's coordinate system:

$$
    P_W \to_{X} P_C \to_{K} p_\text{img}
$$

What are the Depthmaps?

$K$ is a projection with information loss. We can derive the direction, but not the scale. If we backproject a pixel to the camera Coordinate System, we won't get the position of the 3D point but the direction from the camera perspective. Thes directions are called rays.

$$
   Z_c K^{-1} p_{img} = P_C 
$$

Then, depthmaps are comprised of just the third coordinates of the 3d points from its camera point of view.


Confidence maps are meant to be coefficients to reduce the impact of too distanced objects when training. We often want find-grained precision with closer objects, so we weight them more. But appart of being relative coefficients, it's not clear if they actually learn geometric information, theyh seem to be only relative values. So, it does not make same to modify them.


Then, if we want to transform a chunks prediction, we seek the following 3 properties:

- P1. Point maps follow a $Sim(3)$ transformation
$$
    \begin{bmatrix} P_W' \\ 1 \end{bmatrix} = S \begin{bmatrix} P_W \\ 1 \end{bmatrix}
$$

- P2. Images are invariant:

$$
    p_{img}' = p_{img} \\
    p_{img} = \frac{1}{D(u,v)} K P_C \\
    p_{img}' = \frac{1}{D'(u,v)} K' P_C' \\
$$

- P3. Relative positions are constant. Thus:

$$
    \begin{bmatrix} P_C' \\ 1 \end{bmatrix} = S_s \begin{bmatrix} P_C \\ 1 \end{bmatrix} \implies P_C' = sP_C\\
    \begin{bmatrix} P_C' \\ 1 \end{bmatrix} = X \begin{bmatrix} P_W \\ 1 \end{bmatrix} \\
    \begin{bmatrix} P_C' \\ 1 \end{bmatrix} = X' S \begin{bmatrix} P_W \\ 1 \end{bmatrix}
$$

The first equation from the set above tells us that depth maps are only scaled (because depthmaps are the third cooridante of the 3D Points position from camera's pov). That brings us the first statement:

- S1. **Depth maps are scaled**

$$
    D \leftarrow sD
$$

We now will use the facts that image are fixed and relative positions are constant ($P_C' = s P_C$) to find the procedure to update the intrinsics matrix.


$$
 \frac{1}{D(u,v)} K P_C = \frac{1}{D'(u,v)} K' P_C' = \frac{s}{D'(u,v)} K' sP_C \implies K' = K
$$

- S2. **The intrinsics are invariante to $Sim(3)$ transformations**

$$
    K \leftarrow K
$$

Finally, to update the extrinsics:

$$
    \begin{bmatrix} P_C' \\ 1 \end{bmatrix} = S_s \begin{bmatrix} P_C \\ 1 \end{bmatrix}
$$

$$
    X'\begin{bmatrix} P_W' \\ 1 \end{bmatrix} = S_s X \begin{bmatrix} P_W \\ 1 \end{bmatrix}
$$
$$
    X'S \begin{bmatrix} P_W \\ 1 \end{bmatrix} = S_s X \begin{bmatrix} P_W \\ 1 \end{bmatrix} \implies X'= S_sXS^{-1}
$$

- S3. **We use $X' = S_s X S^{-1}$ to update extrinsics**.

### Aligning methods

**1. Point maps alignment**

Recall chunks overlap. So there's a non-empty set of images that belong to both chunk $src$ and chunk $tgt$. We seek for $S^*$ such as

$$
    S^* = \argmin_S ||P_{tgt} - SP_{src} ||
$$

Here $P_{tgt}$ and $P_{src}$ are point maps in homogeneous form.

**2. Aligning shared views**

- Case a) We align only one shared view.

Say we have a shared view $V$ in chunks $src$ and $tgt$. We denote $\{D_{src}, D_{conf_{src}}, X_{src}, K_{src}\}$ to the set of predictions that belong to $V$ in the chunk $i$. Similarly, $\{D_{tgt}, D_{conf_{tgt}}, X_{tgt}, K_{tgt}\}$ belong to $V$ but from chunk $tgt$.

We are looking for the transformation $S$ that converts the extrinsics $X_{src}$ to $X_{tgt}$.

Recall that an extrinsics matrix is transformed as follows:

$$
    X' = S_s X S^{-1}
$$

We intend

$$
    X_{tgt} = S_s X_{src} S^{-1}
$$

Therefore,

$$
    S = X_{tgt}^{-1} S_s X_{src}
$$

Yet $S_s$ is an unknown.

Recall $S_s$ is defined only by the scale factor $s$. We pick $s^*$ as follows:

$$
    s^* = \argmin_s ||D_{tgt} - sD_{src}||
$$

This method is way faster, and we may (FUTURE) incorporate more extrinsics.

## Transforming Poses for visual odometry.

While extrinsics are World-To-Cam transformations, it's possible to derive the position and orientation of a camera from the Cam-To-World Transformation. So, it is usefull to represent the pose of a Camera as the Cam-To-World matrix.
As they are the inverse transformations, the Cam-To-World matrix is the inverse of the extrinsics matrix. We denot with $C = X^{-1}$ the Cam-To-World matrix.

Recall:

$$
    X' = S_s X S^{-1}
$$

Thus

$$
    C' = S C S_s^{-1}
$$

Now, we aim to predict poses relative to the first camera's pose, i.e., $C_1' = I$


$$
    I = S C_1 S_s^{-1} \implies S = S_s C_1^{-1}
$$

Now, how to pick the scale factor $s$ from $S_s$? We could just use $s=1$ and we get metric predictions, but probably the model would perform much better if we use some kind of normalization.

For instance, we can normalize such as the maximum distance is 1.

Then:

$$
    s = \frac{1}{\max ||t_i - t_1||}
$$

But this fails when all views shared the same origin, in other words, when all cameras shared the same position (not the case in either Stereo nor MVS).


Say we have two Views, they poses are:

$$
    C_1 = \begin{bmatrix}
        R_1 & t_1 \\
        0 & 1
    \end{bmatrix}
    \text{ ,}
    C_2 = \begin{bmatrix}
        R_2 & t_2 \\
        0 & 1
    \end{bmatrix}
$$

$$
s = \frac{1}{||t_2 - t_1||}
$$

Now, poses are pre-processed as follows:

$$
    C' = \begin{bmatrix}
            s I & 0 \\
            0 & 1
        \end{bmatrix}
        C_1^{-1}
        C
        \begin{bmatrix}
            \frac{1}{s} I & 0 \\
            0 & 1
        \end{bmatrix} \\
        =
        \begin{bmatrix}
            s I & 0 \\
            0 & 1
        \end{bmatrix}

        \begin{bmatrix}
            R_1^T & -R_1^Tt_1 \\
            0 & 1
        \end{bmatrix}

        \begin{bmatrix}
            R & t \\
            0 & 1
        \end{bmatrix}
    
        \begin{bmatrix}
            \frac{1}{s} I & 0 \\
            0 & 1
        \end{bmatrix} \\ 
    =
        \begin{bmatrix}
            s I & 0 \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            R_1^TR & R_1^T(t - t_1) \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            \frac{1}{s} I & 0 \\
            0 & 1
        \end{bmatrix} \\ 
    = \begin{bmatrix}
            R_1^TR & sR_1^T(t - t_1) \\
            0 & 1
        \end{bmatrix}
$$

Now

$$
    C_2 = 
    \begin{bmatrix}
            R_1^TR_2 & sR_1^T(t_2 - t_1) \\
            0 & 1
        \end{bmatrix}
$$

Thus, 

$$t_2' = sR_1^T(t_2 - t_1)$$

$$
||t_2'|| = s ||t_2 - t_1|| = 1 \implies s = \frac{1}{||t_2 - t_1||}
$$

Never train with less than 2 views.




[DEPRECATED], but includes worth insights

#### Transforming camera extrinsics.

In order to deduce the function to update extrinsics, we need to intoduce extra notation.



When we transform point maps by a global $Sim(3)$ transformation, i.e., $P' = SP$, we must find a extrinsic matrix which satisfies:

$$
    D' = S_{scale} D \\
    P ' = S P \\
    D' = X' P'\\
$$

Where
$$
    \begin{bmatrix}
    D \\ 1
    \end{bmatrix}
    = 
    X P \\
$$

And $X'$ must be a valid extrinsic matrix, i.e., $X' \in SE(3)$

Yet I'm not sure if we must impose some conditions for $S$.

**Possible $Sim(3)$ transformations**

We know that we can express any $SE(3)$ (rigid) transformation where 
$$T = \begin{bmatrix} R & t \\ 0 & 1\end{bmatrix}$$

So we adopt the convention to express every $Sim(3)$ transformation where 
$$
    T = \begin{bmatrix} sR & t \\ 0 & 1\end{bmatrix}
$$

Being $t$ any translation at the scaled and rotated world space, i.e.,

$$
    S = S_t S_R S_s = S_tS_sS_R
$$

Where
$$
    S_s = \begin{bmatrix}
            sI & 0 \\
            0 & 1
        \end{bmatrix} \\
    S_R = \begin{bmatrix}
            R & 0 \\
            0 & 1
        \end{bmatrix} \\
    S_t = \begin{bmatrix}
            0 & t \\
            0 & 1
        \end{bmatrix} 
$$

Note indeed $S_sS_R = S_RS_s$.

Thus, the desired properties can be re-written as a single equation:

$$
 S_{s} X P = X'SP 
$$

Thus

$$
    S_{s} X = X'S \implies X' = S_{s}XS^{-1}
$$

Unfolding the equation above

$$
X' = S_{s} X S_{s}^{-1} S_{R}^{-1}S_t^{-1} \\
    = \begin{bmatrix}
            sI & 0 \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            R_X & t_X \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            s^{-1}I & 0 \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            R^T & 0 \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            I & -t \\
            0 & 1
        \end{bmatrix} \\
        = 
        \begin{bmatrix}
            R_X & st_X \\
            0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            R_T & -R^Tt \\
            0 & 1
        \end{bmatrix} \\
    = \begin{bmatrix}
            R_X & st_X \\
            0 & 1
        \end{bmatrix}
        T^{-1}
$$

Which is equivalent to multiply by $s$ the translation vector of $X$ and then right-multiply by the inverse of the Rigid transformation $[R | t]$.

Following I propose so far two methods to find $s, R, t$, namely, to find the scale change and the rigid transformation.

**- First, and most intuitive**:

We find $S$ which minimizes:

$$
    \text{distance}(\{P_{shared\text{-}views}\}_i, \{SP_{shared\text{-}views}\}_j)
$$

Where distance can be any valid one, e.g., euclidean, $L1$ norm or Huber loss.

**- Second, it's more straightforward, intuitive but not that accurate.**

This is faster, so that's why I implemented it.

First, we find $s$ (with optimization) such as:

$$
    s = \argmin \text{distance}(\{D_{shared\text{-}views}\}_i, \{sD_{shared\text{-}views}\}_j)
$$

Once it is done, we update the extrinsic matrix for this new scaled world.

We must find an extrinsic matrix $X'$ that backprojects a scaled point $S_sP$ to the new depth $S_sD$, in other words, 

$$
    S_s D = X'S_s P
$$

But $D = XP$

Thus

$$
    S_sXP = X'S_sP \implies X' =S_sXS_s^{-1}
$$

Perfect, this is similar to the first part of our function from the point-map-distance minimization.

Now, we want to aign $X'$ with **one and only one** shared-view pose of the destination world.

$$
    X_{i, shared}^{-1} = TX_{j, shared}'^{-1} 
$$ 

By now on, we'll omit the $shared$ sub-index.

Where, $S_RS_t = T$ (any Rigid transformation).

$$
    T = X_{i}^{-1} X_{j}' = X_{i}^{-1} S_sX_{j} S_s^{-1}
$$

So far so good. But look, here we can further simplify it.

Instead of 

$$
    X' = \begin{bmatrix}
            R_X & t_X \\
            0 & 1
        \end{bmatrix}
        T^{-1}
$$

we can write

$$
    X_{j, l}' = S_s X_{j, l} S_s^{-1}T^{-1} =  S_s X_{j, l} S_s^{-1}(X_{i, shared}^{-1} S_sX_{j, shared} S_s^{-1})^{-1} \\
    = S_s X_{j, l} X_{j, shared}^{-1}S_s^{-1}X_{i, shared} 
$$

Note $ X_{j, l}, X_{j, shared}$ shared the same origin. So $X_{j, l} X_{j, shared}^{-1}$ represents the pose of the camera $View_{j, l}$ from the point of view of the camera $View_{j, shared}$

Prove:

$$
    P_{C_l} = X_l (X^{-1}_{s} P_{C_{s}})
$$

Say, there's a transformation which maps one extrinsic (not the poses) to the other. $X_m = T_{ext} X_n$. Therefore, $X_{j, l} X_{j, shared}^{-1}$ is a global $SE(3)$ transformation which converts the extrinsic matrix $X_{j, shared}$ into the other extrinsic matrix into $X_{j, l}$. Yeah, that's odd.

Ok, let's work with poses instead.

$$
    X_{j, l}'^{-1} = X_{i, shared}^{-1} S_s X_{j, shared} X_{j, l}^{-1} S_s^{-1}
$$


Let $X_0$ and $X_1$ be extrinsics of the same view but from different chunks. We aim to find a procedure that aligns chunk 0 and chunk 1 such as $X_1' = X_0$. We do have the depth maps $D_0$ and $D_1$ so we may easily find $s^*$ such as $||D_0 - sD_1||_{Huber}$ is minimum.

$$
    sD = S_{scale}D_{homo} = S_{scale} X P = X'T S_{scale} P
    \\
    \implies \\
    S_{scale}X = X'TS_{scale} \\
    \implies \\
    X' = S_{scale} X S_{scale}^{-1} T^{-1}
$$

Where $S_{scale} = \begin{bmatrix} sI & 0 \\ 0 & 1\end{bmatrix}$, $s$ is such that $||D_0 - sD_1||_{Huber}$ is minimum.  And $X'$ is a valid extrinsic matrix since $S_{scale} X S_{scale}^{-1}$ affects only the translationn vector of $X$.

But, what about $T$? Say we have a shared view $V_{shared}$ in both chunks. They, somehow must be the same. The poses of both chunks build a trajectory. The shared views must (ideally) build the same trajectory, but scaled and transformed. However, if only one view is available, it's difficult to find the $Sim(3)$ transformation. We first scale.
