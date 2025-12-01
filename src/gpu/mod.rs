use wgpu::*;
use pollster::block_on;

pub struct GpuBackend {
    pub device: Device,
    pub queue: Queue,
}

impl GpuBackend {
    pub fn new() -> Self {
        block_on(async {
            let instance = Instance::default();

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                })
                .await
                .expect("No compatible GPU found");

            let (device, queue) = adapter
                .request_device(
                    &DeviceDescriptor {
                        label: None,
                        features: Features::empty(),
                        limits: Limits::default(),
                    },
                    None,
                )
                .await
                .expect("Failed to create GPU device");

            Self { device, queue }
        })
    }
}
