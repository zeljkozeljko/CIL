import numpy
from numpy.linalg import norm
import os
import sys
import shutil
import unittest
from ccpi.framework import BlockDataContainer
try:
    import sirf.STIR as pet
    from sirf.Utilities import  examples_data_path
    has_sirf = True
except ImportError as ie:
    has_sirf = False

class UnsupportedObject(object):
    pass

class SupportedObject(object):
    def __rmul__(self, other):
        return True
    def __radd__(self, other):
        return True
    def __rsub__(self, other):
        return True

class TestSIRFCILIntegration(unittest.TestCase):
    
    def setUp(self):
        if has_sirf:
            os.chdir(examples_data_path('PET'))
            # Copy files to a working folder and change directory to where these files are.
            # We do this to avoid cluttering your SIRF files. This way, you can delete 
            # working_folder and start from scratch.
            shutil.rmtree('working_folder/brain',True)
            shutil.copytree('brain','working_folder/brain')
            os.chdir('working_folder/brain')

            self.cwd = os.getcwd()

    
    def tearDown(self):
        if has_sirf:
            shutil.rmtree(self.cwd)
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_SIRF_DataContainer_max(self):
        print("test SIRF DataContainer max")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image1.fill(1.)
        arr = image1.as_array()
        arr[0][0][0] = 10
        image1.fill(arr)
        assert image1.max() == 10
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_SIRF_DataContainer_dtype(self):
        print("test SIRF DataContainer max")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        assert image1.dtype == numpy.float32

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_divide(self):
        print ("test_BlockDataContainer_with_SIRF_DataContainer_divide")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        print (image1.shape, image2.shape)
        
        tmp = image1.divide(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.divide(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.divide(1.)

        self.assertBlockDataContainerEqual(bdc , bdc1)

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_multiply(self):
        print("test_BlockDataContainer_with_SIRF_DataContainer_multiply")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(1.)
        image2.fill(2.)
        print (image1.shape, image2.shape)
        
        tmp = image1.multiply(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        tmp = image2.multiply(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        

        # image.fill(1.)
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.multiply(1.)

        self.assertBlockDataContainerEqual(bdc , bdc1)
    
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_add(self):
        print("test_BlockDataContainer_with_SIRF_DataContainer_add")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(0)
        image2.fill(1)
        print (image1.shape, image2.shape)
        
        tmp = image1.add(1.)
        numpy.testing.assert_array_equal(image2.as_array(), tmp.as_array())
        tmp = image2.subtract(1.)
        numpy.testing.assert_array_equal(image1.as_array(), tmp.as_array())
        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.add(1.)

        image1.fill(1)
        image2.fill(2)

        bdc = BlockDataContainer(image1, image2)

        self.assertBlockDataContainerEqual(bdc , bdc1)
    
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_BlockDataContainer_with_SIRF_DataContainer_subtract(self):
        print("test_BlockDataContainer_with_SIRF_DataContainer_subtract")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        image2 = pet.ImageData('emission.hv')
        image1.fill(2)
        image2.fill(1)
        print (image1.shape, image2.shape)
        
        bdc = BlockDataContainer(image1, image2)
        bdc1 = bdc.subtract(1.)

        image1.fill(1)
        image2.fill(0)

        bdc = BlockDataContainer(image1, image2)

        self.assertBlockDataContainerEqual(bdc , bdc1)

    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_SIRF_DataContainer_multiply_with_other_object(self):
        print("test_SIRF_DataContainer_multiply_with_other_object")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        unsupported = UnsupportedObject()
        supported = SupportedObject()

        try:
            ret = image1 * unsupported
            self.assertTrue(False)
        except TypeError as te:
            print ("Catching ", te)
            self.assertTrue(True)
        
        ret = image1 * supported
        self.assertTrue(ret)
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_SIRF_DataContainer_add_with_other_object(self):
        print("test_SIRF_DataContainer_multiply_with_other_object")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        unsupported = UnsupportedObject()
        supported = SupportedObject()

        try:
            ret = image1 + unsupported
            self.assertTrue(False)
        except TypeError as te:
            print ("Catching ", te)
            self.assertTrue(True)
        
        ret = image1 + supported
        self.assertTrue(ret)
    
    @unittest.skipUnless(has_sirf, "Has SIRF")
    def test_SIRF_DataContainer_subtract_with_other_object(self):
        print("test_SIRF_DataContainer_multiply_with_other_object")
        os.chdir(self.cwd)
        image1 = pet.ImageData('emission.hv')
        unsupported = UnsupportedObject()
        supported = SupportedObject()

        try:
            ret = image1 - unsupported
            self.assertTrue(False)
        except TypeError as te:
            print ("Catching ", te)
            self.assertTrue(True)
        
        ret = image1 - supported
        self.assertTrue(ret)

    def assertBlockDataContainerEqual(self, container1, container2):
        print ("assert Block Data Container Equal")
        self.assertTrue(issubclass(container1.__class__, container2.__class__))
        for col in range(container1.shape[0]):
            if hasattr(container1.get_item(col), 'as_array'):
                print ("Checking col ", col)
                self.assertNumpyArrayEqual(
                    container1.get_item(col).as_array(), 
                    container2.get_item(col).as_array()
                    )
            else:
                self.assertBlockDataContainerEqual(container1.get_item(col),container2.get_item(col))
    
    def assertNumpyArrayEqual(self, first, second):
        numpy.testing.assert_array_equal(first, second)
