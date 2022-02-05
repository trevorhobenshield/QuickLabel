# QuantLabel

Parallel dataset auto-labeling using SOTA quantized (PTQ) tflite models

## Example

`label-interactive.ipynb`

<br>

Given the file `keywords.txt`

```
seashore
mobile home
soccer ball
```

<br>


Networks perform inference on the data inside `images/`. User is notified as to which classes have been found, if any.

<br>


![](resources/found_images.png)

![](resources/df.png)

<br>


Times are shown below, in case you want to swap out slower/faster networks.

<br>


![](resources/times.png)

<br>


Final output is shown. Directories are created for each class that was found, which match your keywords
in `keywords.txt`. Images are moved into their respective class-labeled directories.

<br>

![](resources/output.png)
