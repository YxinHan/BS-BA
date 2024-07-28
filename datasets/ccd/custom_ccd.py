import os.path as osp
from functools import reduce

import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import scd_eval_metrics,cd_eval_metrics
from ..builder import DATASETS
from ..custom_cd import CustomDatasetCD
from ..pipelines import ComposeWithVisualization
import mmcv

from osgeo import gdal
@DATASETS.register_module()
class CustomDatasetCCD(CustomDatasetCD):
    '''
    Base class for datasets for Conditional CD.
    '''
    def __init__(self,
                 pipeline,
                 img1_dir,
                 img2_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index_bc=255,
                 ignore_index_sem=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 if_visualize=True,
                 ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img1_dir):
                self.img1_dir = osp.join(self.data_root, self.img1_dir)
                self.img2_dir = osp.join(self.data_root, self.img2_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img1_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric: Dummy argument for compatibility.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        gt_sem_maps = self.get_gt_sem_maps(efficient_test)
        gt_bc_maps = self.get_gt_bc_maps(efficient_test)
        # print(gt_bc_maps,np.amax(gt_bc_maps))
        # print(gt_sem_maps,np.amax(gt_sem_maps))
        if self.CLASSES is None:
            num_semantic_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_sem_maps]))
        else:
            num_semantic_classes = len(self.CLASSES)

        ret_metrics = scd_eval_metrics(
            results=results,
            gt_bc_maps=gt_bc_maps,
            gt_sem_maps=gt_sem_maps,
            num_semantic_classes=num_semantic_classes,
            ignore_index_bc=self.ignore_index_bc,
            ignore_index_sem=self.ignore_index_sem
        )

        if self.CLASSES is None:
            class_names = tuple(range(num_semantic_classes))
        else:
            class_names = self.CLASSES

        BCD_metrics = ['BC_precision', 'BC_recall', 'BC','SC', 'SCS', 'mIoU']
        summary_table = PrettyTable(field_names=BCD_metrics)
        summary_table.add_row([np.round(ret_metrics[m], decimals=3) for m in BCD_metrics])

        # SCD_metrics = []
        # summary_table = PrettyTable(field_names=SCD_metrics)
        # summary_table.add_row([np.round(ret_metrics[m], decimals=3) for m in SCD_metrics])

        print_log('Summary:', logger=logger)
        print_log('\n' + summary_table.get_string(), logger=logger)

        classwise_table = PrettyTable(field_names=['Class'] + list(class_names))
        classwise_table.add_row(['IoU'] + list(np.round(ret_metrics['IoU_per_class'], decimals=3)))
        classwise_table.add_row(['SC'] + list(np.round(ret_metrics['SC_per_class'], decimals=3)))

        print_log('per class results:', logger=logger)
        print_log('\n' + classwise_table.get_string(), logger=logger)




        # img = mmcv.imread(img)  # (h, w, 3)
        # img = img.copy()
        # for i in range(len(results)):
        #     seg_sem = results[i]['bc']
        #     # seg_bc = results[i]['bc']
        # # seg = result[0]  # seg.shape=(h, w). The value in the seg represents the index of the palette.
        #
        #     palette = np.random.randint(
        #         0, 255, size=(len(self.CLASSES), 3))
        #
        #     palette = np.array(palette)
        #     assert palette.shape[0] == len(self.CLASSES)
        #     assert palette.shape[1] == 3
        #     assert len(palette.shape) == 2
        #     # assert 0 < opacity <= 1.0
        #     color_seg = np.zeros((seg_sem.shape[0], seg_sem.shape[1], 3), dtype=np.uint8)  # (h, w, 3). Drawing board.
        #     for label, color in enumerate(palette):
        #         color_seg[seg_sem == label,
        #         :] = color  # seg.shape=(h, w). The value in the seg represents the index of the palette.
        #     # convert to BGR
        #     color_seg = color_seg[..., ::-1]
        #
        # img = color_seg
        # img = img.astype(np.uint8)
        # # if out_file specified, do not show image in window
        #
        #
        # mmcv.imshow(img,'', 0)




            # if dataset is None:
            #     raise Exception("Unable to open the input TIFF file.")

        for i in range(len(results)):
            # for site_pre in self.sites:
                 site = self.sites
                 seg_sem = results[i]['bc']
                 # image_array = np.array(seg_sem.GetRasterBand(1).ReadAsArray())
                 tif = self.img_suffix
                 path = "./data/HRSCD/res"
        
                 output_path = osp.join(path, site[i] + tif)
        
                 filename = osp.join(self.img_dir, '2006', site[i] + tif)
        
                 # print(filename,output_path)
        
                 dataset = gdal.Open(filename, gdal.GA_ReadOnly)
                 # print(dataset.GetGeoTransform())
                 image_array = np.array(dataset.GetRasterBand(1).ReadAsArray())
        # 将像素值限制在 0 到 4 之间
        #          clamped_array = np.clip(image_array, 0, 1)
        
        # 创建一个新的 TIFF 文件
                 driver = gdal.GetDriverByName("GTiff")
                 clamped_dataset = driver.Create(output_path,256,256, 1, gdal.GDT_Float32)
        
        # 设置地理参考信息
                 clamped_dataset.SetGeoTransform(dataset.GetGeoTransform())
                 clamped_dataset.SetProjection(dataset.GetProjection())
        
        # 将 NumPy 数组写入新的 TIFF 文件
                 clamped_dataset.GetRasterBand(1).WriteArray(seg_sem)
        
        # 释放资源
                 dataset = None
                 clamped_dataset = None
        # #


        return ret_metrics