Well, this is a Computer Vision 101 class' topic, so why bother to discusse it? Answer. Most of the time I forgot the physical aspects of them.

Still, the pinhole perspective (also called central perspective) projection model, first proposed by Brunelleschi at the beginning of the fifteen century, is mathematically convenient and it often provides an acceptable approximation of the imaging process.

Some notation:

- $\Pi$: image plane.
- Optical axis: The line perpendicular to $\Pi$ and passing through the pinhole.
- Image center. The point $c$ where the optical axis pierces $\Pi$.

Within the pinhole projection model, we set a coordinate system $(O, i, j, k)$ whose origin $O$ coincides with the pinhole. The image plane is located at a positive distance $d$ from the pinhole along the vector $k$.

Let $P$ denote a scene point with coordinates $(X, Y, Z)$ and $p$ denote its image with coordinates $(x, y, z)$. We have:

$$
x = \lambda X \\
y = \lambda Y \\
z = d = \lambda Z
$$

$d$ is a fixed value, so
$$
    x = d\frac{X}{Z} \\
    y = d\frac{Y}{Z}
$$

**Cameras with Lenses**
Lenses are added for two main reasons:
- To gather light. Small pinholes gather little light. Larger pinholes produce bright, albeit blurry, images.
- Keeping the picture in sharp focus while gathering light from a large area.

When working with camera models we often ignore diffraction, interferences and other physical optics phenomena. Doing so, the behavior of lenses is dictated by the laws od geometric optics. Actually, only considering refraction suffices to describe the imaging process (this does not hold for catadioptric optical systems, e.g., telescopes).

Refraction is well described by Snell's law:

$$
    n_1 \sin \alpha_1 = n_2 \sin \alpha_2
$$

Where $n_1, n_2$ are the indices of refraction of different mediums (e.g., the lens itself and air or vacuum); $\alpha_1$ is the incidence angle and $\alpha_2$ is the refracted angle.

For a camera model with out aberrations, we consider the first order \ expantion of the sine function is a good approximation, i.e., $n_1 \alpha_1 \approx n_2 \alpha_2$



- **Thin lens model**

$$
    \frac{1}{z} - \frac{1}{Z} = \frac{1}{f}
$$

where $f = \frac{R}{2(n-1)}$, being $R$ the radius of the spherical surfaces of the lens. Also, we consider that the refraction index of air equals to 1. Also, note that, in this form, we expect $Z \leq 0$.

If $Z \to - \infty$, then $z = f$. Then, the objects at a $-\infty$ distance are focused in the image plane at a distance $f$. The two points $F$ and $F'$ located at distance $f$ from the lens center on the optical center are called the focal points of the lens.



- **Field of View**

    The portion of scene space that actually projects onto the retina of the camera.
    It depends on the focal lenght and the effective area of the retina. It can be defined as $2 \phi$, whre $\arctan \frac{a}{2f}$, $a$ is the diameter of the sensor.

Spherical aberration happens when the $\sin \alpha \approx \alpha$ is not a good approximation. We'll see that a scene point $P$ does not collide into a single image point $p$, but into a circle. There are four other types of primary aberrations caused by the differences between first- and third-order optics, namely *coma*, *astigmatism*, *field curvature*, and *distortion*. The first three adds blurring while the *distortion* changes the shape of the image as a whole.

**The index refraction of a transparent medium depends on wavelentgh**, so does focal length. This causes the penomenon of *chromatic aberration*.

Aberrations are minimized by aligning several simple lenses qith well-chosend shapes and refraction indices, separated by appropiate stops. These lense systems can still be modeled as a sinlge thin len.


## Intrinsics and Extrinsics.

We now consider a *normalized image plane* (where $f = 1$) parallel to its physical retina but located at a unit distance from the pinhole. We attach to this plane its own coordinate system with an origin located at the point $c$ where the optical axis pierces it. Due to the normalization, now we have

$$
    \hat x = \frac{X}{Z} \\
    \hat y = \frac{Y}{Z}
$$

Thus

$$
    \hat p = \frac{1}{Z}
    \begin{bmatrix}
        I & 0
    \end{bmatrix}
    P
$$

Where $p = (\hat x, \hat y, 1)^T$ are the coordinates of the projection in the normalized image plane, and $P = (X, Y, Z, 1)$ is a homogeneous vector of the projecting point in the world coordinate frame.

**It turns out that, in general, the physical retina is not at a distance $f = 1$, while the normalized image plane is.** Does this change the equations?

The image coordinates are not expressed in either the normalized image coordinates nor meters, but expressed in pixel units; the origin of the image coordinate system is a corner. In addition, **pixels are rectangular instead of square (not the image?)**. 

All right, let's derive the Intrinsic matrix

Say $\hat x, \hat y$ are coordinates of the normalized image plane. A pixel has dimensions $\frac{1}{k} \times \frac{1}{l} [\frac{pixel^2}{m^2}]$, we have.

$$
    x = kf\frac{X}{Z} = k f \hat x = \alpha \hat x \\
    y = lf\frac{X}{Z} = l f \hat y = \beta \hat y
$$

All, right. Here $f$ (the focal length) plays as a scale ratio to convert normalized coordinates to metric. 

But wait, an image center is at a corner. We must add an offset.

$$
    x =  \alpha \hat x + x_0\\
    y =  \beta \hat y + y_0
$$

Finally, a common manufacturing error might skew the coordinate system, so the angle $\theta$ among the two image axis is not equal (but not very different from) 90 degres.

$$
x = \alpha \hat x - \alpha \cot \theta \hat y + x_0 \\
y = \frac{\beta}{\sin \theta} + y_0
$$

Note $x_0, y_0$ units are pixels. But, what about $\hat x, \hat y$? Are they still noormalized coorditnates? Why not?

Indeed, $x = \frac{X}{Z}$, $y = \frac{Y}{Z}$. So, no problem. On the other hand, $\alpha = kf \left[\frac{1}{m} \cdot  m\right]$. So, once again, there are no units at all in these equation.

Ok, now fe got a form of the intrinsics matrix.

$$
p = \mathcal K \hat p
$$

Where $p = (x, y, 1)^T$ and 

$$
    \mathcal K = \begin{pmatrix}
            \alpha & - \alpha \cot \theta & x_0 \\
            0 & \frac{\beta}{\sin \theta} & y_0 \\
            0 & 0 & 1
        \end{pmatrix}
$$

Well, here $\mathcal K$ is the internal calibration matrix.

Yet, something is odd. This formula expects inputs from a normalized ($f=1$) coordinate system. Using points from a real-world coordinate (still from camera perspective):

$$
    p = \frac{1}{Z} \mathcal K 
    \begin{bmatrix}
        I, 0
    \end{bmatrix}
    P
$$

Several of this intrinsic camera parameters, such as the focal length, or the physical size of the pixels, are often available in the EXIF tags attached to the JPEG images.

This model also ignores zoom lenses, which have the capability to vary the focal length.

Finally (To be elaborated later), we include the extrinsic matrix and we can project a 3D point from any scene coordinate frame (no need to be attached to the camera) to the image (in pixel units) coordinate system.

$$
    p = \frac{1}{Z_c} \mathcal K
        \begin{bmatrix}
             R & t 
        \end{bmatrix}
        P_w
$$

But you know, the scaling to $\frac{1}{Z}$ is used just to ensure the last element of $p$ equals 1. In practice, we can omit this factor and apply it when computing the pixel coordinates. This is true because the following holds:

$$
\begin{bmatrix}
    u & v & 1
\end{bmatrix}^T
= 
\begin{bmatrix}
    Zu & Zv & Z
\end{bmatrix}^T
=
Z
\begin{bmatrix}
    u & v & 1
\end{bmatrix}^T
$$

For every $z \in R$.
