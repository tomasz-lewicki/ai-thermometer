FACE_BB_COLOR = (255, 255, 255) # white
EYES_BB_COLOR = (0,   255, 255) # yellow


def overlay_bboxes(arr, detections, temps):

    for (face, eyes), temp in zip(detections, temps):

        print(face)
        x, y, w, h = face

        # draw face bounding box
        cv2.rectangle(arr, (x, y), (x + w, y + h), FACE_BB_COLOR, 2)

        deg_c = round(temp['face'], 2)
        deg_f = round(ctof(temp['face']), 2)

        # draw facial temperature
        cv2.putText(
            arr,
            text=f"{deg_c} deg C {deg_f} deg F",
            org=(x,y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2
        )

        roi = arr[y : y + h, x : x + w]

        for (ex, ey, ew, eh), eye_t in zip(eyes, temp['eyes']):
            # draw eye bouding box
            cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), EYES_BB_COLOR, 2)

    return arr