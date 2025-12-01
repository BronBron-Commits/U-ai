use wgpu::*;
use wgpu::util::DeviceExt;

pub fn matmul_gpu(
    backend: &crate::gpu::GpuBackend,
    a: &[f32],
    b: &[f32],
    size: usize,
) -> Vec<f32> {
    let device = &backend.device;
    let queue = &backend.queue;

    let buffer_size = (size * size * 4) as u64;

    let a_buf = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(a),
        usage: BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(b),
        usage: BufferUsages::STORAGE,
    });

    let c_buf = device.create_buffer(&BufferDescriptor {
        label: Some("C"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(include_wgsl!("matmul.wgsl"));

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Matmul Pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder =
        device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    {
        let mut pass =
            encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size as u32 / 16, size as u32 / 16, 1);
    }

    queue.submit(Some(encoder.finish()));

    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder2 =
        device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder2.copy_buffer_to_buffer(&c_buf, 0, &readback, 0, buffer_size);
    queue.submit(Some(encoder2.finish()));

    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, |_| ());
    device.poll(Maintain::Wait);

    let data = slice.get_mapped_range();

    let result = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    readback.unmap();

    result
}
