use ash::{
    version::{EntryV1_0, InstanceV1_0, DeviceV1_0},
    vk,
};
use std::ffi::CString;

pub struct VulkanBackend {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
}

impl VulkanBackend {
    pub fn new() -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        // Instance
        let app_name = CString::new("u-ai").unwrap();
        let engine_name = CString::new("uai-engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        // Physical device selection (pick the AMD one)
        let pdevices = unsafe { instance.enumerate_physical_devices().unwrap() };

        let physical_device = pdevices
            .into_iter()
            .find(|pd| {
                let props = unsafe { instance.get_physical_device_properties(*pd) };
                props.vendor_id == 0x1002 // AMD
            })
            .expect("AMD GPU not found");

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .unwrap()
        };

        let queue_priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority)
            .build()];

        let device_create = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);

        let device =
            unsafe { instance.create_device(physical_device, &device_create, None).unwrap() };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Self {
            entry,
            instance,
            device,
            queue,
            queue_family_index,
        }
    }
}
