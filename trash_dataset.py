import os

from mrcnn.utils import Dataset


class TrashDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "trash")
        # define data locations
        imgs = list(c.imgToAnns.values())
        if is_train:
            imgs = imgs[:300]
        else:
            imgs = imgs[300:]
        for img in imgs:
            # extract image id
            image_id = img[0]['image_id']
            image_data = c.loadImgs([image_id])[0]
            segmentations = []
            for i in img:
                x = []
                y = []
                for idx, v in enumerate(i['segmentation'][0]):
                    if idx % 2 ==0:
                        x.append(v)
                    else:
                        y.append(v)
                segmentations.append({'all_points_x': x, 'all_points_y': y})

            image_path = dataset_dir + image_data['file_name']
            height = image_data['height']
            width = image_data['width']        
            if os.path.exists(image_path):
                self.add_image(
                    "dataset",
                    image_id=image_data['file_name'],
                    path=image_path,
                    width=image_data['width'],
                    height=image_data['height'],
                    polygons=segmentations
                )
            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
            Returns:
            masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "dataset":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "database":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
