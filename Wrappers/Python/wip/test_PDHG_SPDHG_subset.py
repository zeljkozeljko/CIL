from ccpi.astra.operators import AstraProjectorSimple
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
from ccpi.optimisation.algorithms import SPDHG, PDHG
from ccpi.optimisation.algorithms import SPDHGFactory
from ccpi.framework import TestData
import numpy as np
from ccpi.utilities.display import plotter2D



# class SubsetKullbackLeibler(KullbackLeibler):
#     def __init__(self, A, b, c=1.0):
#         super(SubsetKullbackLeibler, self).__init__(A, b, c)
        
#     def notify_new_subset(self, subset_id, number_of_subsets):
#         self.b.geometry.subset_id = subset_id
#         self.A.notify_new_subset(subset_id, number_of_subsets)
class AstraSubsetProjectorSimple(AstraProjectorSimple):
    
    def __init__(self, geomv, geomp, device, **kwargs):
        kwargs = {'indices':None, 
                  'subset_acquisition_geometry':None,
                  #'subset_id' : 0,
                  #'number_of_subsets' : kwargs.get('number_of_subsets', 1)
                  }
        # This does not forward to its parent class :(
        super(AstraSubsetProjectorSimple, self).__init__(geomv, geomp, device)
        number_of_subsets = kwargs.get('number_of_subsets',1)
        # self.sinogram_geometry.generate_subsets(number_of_subsets, 'random')
        if geomp.number_of_subsets > 1:
            self.notify_new_subset(0, geomp.number_of_subsets)
        self.is_subset_operator = True

        
    def notify_new_subset(self, subset_id, number_of_subsets):
        # print ('AstraSubsetProjectorSimple notify_new_subset')
        # updates the sinogram geometry and updates the projectors
        self.subset_id = subset_id
        self.number_of_subsets = number_of_subsets

        device = self.fp.device
        # this will only copy the subset geometry
        # it relies on the fact that we are using a reference of the
        # range geometry which gets modified outside.
        # this is rather dangerous!!!
        ag = self.range_geometry().copy()
        #print (ag.shape)
        
        self.fp = AstraForwardProjector(volume_geometry=self.domain_geometry(),
                                        sinogram_geometry=ag,
                                        proj_id = None,
                                        device=device)

        self.bp = AstraBackProjector(volume_geometry = self.domain_geometry(),
                                        sinogram_geometry = ag,
                                        proj_id = None,
                                        device = device)
    

    
    def select_subset(self, subset_id, num_subsets):
        '''alias of notify_new_subset'''
        print ("Should select subset")
        self.notify_new_subset(subset_id, num_subsets)

    @property
    def num_subsets(self):
        return self.range_geometry().number_of_subsets

class SubsetKullbackLeibler(KullbackLeibler):
    def __init__(self, **kwargs):
        super(SubsetKullbackLeibler, self).__init__(**kwargs)
        
    @property
    def is_subset_function(self):
        return True
        
    def select_subset(self, subset_id, number_of_subsets):
        self.b.geometry.subset_id = subset_id
        self.eta.override_subsets(self.b.geometry)
        print ("Called select_subset of SubsetKullbackLeibler")
    
        
        

loader = TestData()
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
print ("here")
ig = data.geometry
ig.voxel_size_x = 0.1
ig.voxel_size_y = 0.1
    
detectors = ig.shape[0]
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
# Select device
# device = input('Available device: GPU==1 / CPU==0 ')
# if device=='1':
#     dev = 'gpu'
# else:
#     dev = 'cpu'
dev = 'gpu'

Aop = AstraProjectorSimple(ig, ag, dev)

sin = Aop.direct(data)
# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[1]
noisy_data = sin.geometry.allocate(None)
    
if noise == 'poisson':
    np.random.seed(10)
    scale = 5
    eta = 0
    noisy_data.fill(np.random.poisson( scale * (eta + sin.as_array()))/scale)
    
elif noise == 'gaussian':
    np.random.seed(10)
    n1 = np.random.normal(0, 0.1, size = ag.shape)
    noisy_data.fill(n1 + sin.as_array())
    
else:
    raise ValueError('Unsupported Noise ', noise)

#%% 'explicit' SPDHG, scalar step-sizes
physical_subsets = 10
num_subsets = physical_subsets
size_of_subsets = int(len(angles)/num_subsets)
# create Gradient operator
op1 = Gradient(ig)
if True:
    ### create subsets
    # noisy_data.generate_subsets(physical_subsets, 'stagger')
    alpha = 0.5
    skl = SubsetKullbackLeibler(b = noisy_data)
    F = BlockFunction(skl, alpha * MixedL21Norm())
    G = IndicatorBox(lower=0)
    # probabilities 1/2 projections 1/2 regularisation
    A = AstraSubsetProjectorSimple(ig, ag, device='gpu')
    K = BlockOperator(A, op1)
    
    algo = SPDHGFactory.get_algorithm(F, G, K, tau=None, sigma=None, \
                      x_init=None, prob=None, gamma=1., norms=None,\
                      max_iteration=1000, update_objective_interval=100,
                      num_physical_subsets=physical_subsets,
                      physical_subsets_method='stagger', data=noisy_data)
    print (algo.get_last_objective())
    algo.run(1000, verbose=False)
    plotter2D(algo.get_output(), cmap='viridis')
    
else:
    # take angles and create uniform subsets in uniform+sequential setting
    list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
    # create acquisitioin geometries for each the interval of splitting angles
    list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
    for i in range(len(list_angles))]
    # create with operators as many as the subsets
    A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) \
        for i in range(num_subsets)] + [op1])
    ## number of subsets
    #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
    #
    ## acquisisiton data
    g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:]) for i in range(0, len(angles), size_of_subsets)])
    alpha = 0.5
    ## block function
    F = BlockFunction(*[*[KullbackLeibler(b=g[i]) for i in range(num_subsets)] + [alpha * MixedL21Norm()]]) 
    G = IndicatorBox(lower=0)

    print ("here")
    prob = [1/(2*num_subsets)]*(len(A)-1) + [1/2]

# norms = [A[i].norm() for i in range(len(A))]
# rho = 0.99
# gamma = 1
# finfo = np.finfo(dtype = np.float32)

# sigma = [rho * gamma / ni for ni in norms]
# sigma = [si * (1 - 1000 * finfo.eps) for si in sigma]
# sigma = [1e-3 for _ in sigma]
sigma = None

algos = []
algos.append( SPDHG(f=F,g=G,operator=A, 
            max_iteration = 1000000,
            update_objective_interval=100, prob = prob, use_axpby=True,
            sigma = sigma)
)
algos[0].run(1000, very_verbose = True)

dA = BlockOperator(op1, Aop)
dF = BlockFunction(alpha * MixedL21Norm(), KullbackLeibler(b=noisy_data))
algos.append( PDHG(f=dF,g=G,operator=dA, 
            max_iteration = 1000000,
            update_objective_interval=100, use_axpby=True)
)
algos[1].run(1000, very_verbose = True)


# np.testing.assert_array_almost_equal(algos[0].get_output().as_array(), algos[1].get_output().as_array())
from ccpi.utilities.quality_measures import mae, mse, psnr
qm = (mae(algos[0].get_output(), algos[1].get_output()),
    mse(algos[0].get_output(), algos[1].get_output()),
    psnr(algos[0].get_output(), algos[1].get_output())
    )
print ("Quality measures", qm)
plotter2D([algos[0].get_output(), algos[1].get_output()], titles=['axpby True eps' , 'axpby True'], cmap='viridis')