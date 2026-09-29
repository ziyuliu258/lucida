#!/usr/bin/env python3
"""Build auditable videos from saved successful GizmoAct evaluation observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import textwrap
from pathlib import Path

import av
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps

from lucida_mini.actions import Stop, parse_action
from lucida_mini.evaluation import full_loop_success
from lucida_mini.expert import execute_action
from lucida_mini.metrics import add_sb, object_diameter, rotation_geodesic_deg, transform_normalized_points
from lucida_mini.pose import Pose
from lucida_mini.schema import DatasetManifest
from lucida_mini.serialization import action_to_text
from lucida_mini.state import GizmoState

DEFAULT_IDS = ['foundationpose_01_08', 'foundationpose_02_04', 'front_01_02', 'front_01_08', 'front_02_08', 'ca1m_01_07']
NAMES = {'foundationpose_01': '恐龙玩具', 'foundationpose_02': '黑色靴子', 'front_01': '手摇咖啡磨', 'front_02': '绿色笔', 'ca1m_01': '卓别林海报'}
SIZE = (1600, 1000)
BG, PANEL, INK, MUTED = '#101827', '#1b2638', '#edf3fb', '#abbcd0'
BLUE, ORANGE, GREEN = '#70b5ff', '#ffbe68', '#67ddb1'


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def pose(record):
    return Pose(np.asarray(record.position_m), np.asarray(record.rotation_object_to_world), np.asarray(record.size_m))


def pose_record(value):
    return {'position_m': value.position.tolist(), 'rotation_object_to_world': value.rotation.tolist(), 'size_m': value.size.tolist()}


def action_label(record):
    if record is None:
        return 'Initial state'
    if record['source'] == 'injected_error':
        return 'Preset perturbation'
    name = record['action'].split('>')[0][1:]
    return {'update_pose': 'Update pose', 'switch_obs': 'Switch view', 'permute_axis': 'Permute axes', 'stop': 'Stop / success'}[name]


def action_details(record):
    if record is None:
        return ['Before the first action.']
    raw = record['action']
    if raw.startswith('<switch_obs>'):
        return ['Switch to six local-axis views.', 'Object pose stays unchanged.']
    if raw.startswith('<permute_axis>'):
        params = json.loads(raw.split('>', 1)[1].split('<')[0])
        return ['Axis mapping: ' + ', '.join(f'{k} <- {v}' for k, v in params.items())]
    if raw.startswith('<stop>'):
        return ['Model outputs STOP.', 'Pose stays unchanged.', 'The final observation is held.']
    params = json.loads(raw.split('>', 1)[1].split('<')[0])
    lines = []
    for key, label, axes in [('rotate', 'Rotate Z/X/Y (deg)', 'zxy'), ('translate', 'Translate X/Y/Z (local fraction)', 'xyz'), ('scale', 'Scale X/Y/Z (fraction)', 'xyz')]:
        if key in params:
            lines += [label, ' / '.join(f'{params[key][axis]:+.2f}' for axis in axes)]
    return lines


def paste_fit(canvas, path, box):
    x, y, w, h = box
    with Image.open(path) as original:
        frame = ImageOps.contain(original.convert('RGB'), (w, h), Image.Resampling.LANCZOS)
    canvas.paste(frame, (x + (w - frame.width) // 2, y + (h - frame.height) // 2))


def observation_paths(directory, index):
    stem = f'step_{index:02d}'
    single = directory / (stem + '.png')
    separate = [single, directory / f'{stem}_overlay.png', directory / f'{stem}_pointcloud.png']
    orthographic = [
        directory / f'{stem}_local_{axis}_{sign}.png'
        for axis in ('x', 'y', 'z')
        for sign in ('pos', 'neg')
    ]
    if all(path.exists() for path in separate):
        return separate + (orthographic if all(path.exists() for path in orthographic) else [])
    if single.exists():
        return [single]
    paths = [directory / f'{stem}_{suffix}.png' for suffix in ['reference', 'current', 'focus']]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f'Missing observations for {directory}/{stem}')
    return paths


def draw_frame(tid, context, index, records, metrics, obs, mode, checkpoint, presentation_final_scene=False):
    canvas = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(canvas)
    n = len(records)
    record = records[index - 1] if index else None
    color = GREEN if index == n else ORANGE if record and record['source'] == 'injected_error' else BLUE
    d.text((26, 18), 'LUCIDA / GIZMOACT', font=font(31, True), fill=INK)
    d.text((26, 60), f'{tid}   |   {context.target_category}   |   checkpoint {checkpoint}', font=font(22), fill=MUTED)
    status = 'SUCCESS' if index == n else 'INITIAL' if index == 0 else f'AFTER ACTION {index}'
    d.text((1090, 28), status, font=font(31, True), fill=color)
    d.rounded_rectangle((20, 111, 1042, 964), 14, fill=PANEL)
    view_title = 'FINAL SOURCE SCENE / POSE UNCHANGED' if presentation_final_scene else 'SAVED MODEL OBSERVATION  /  ' + mode.upper().replace('_', ' ')
    d.text((36, 123), view_title, font=font(20, True), fill=GREEN if presentation_final_scene else MUTED)
    if len(obs) >= 3:
        labels = ('RAW RGB', 'MATERIAL OVERLAY', 'RGB-COLORED POINT CLOUD')
        for index, (path, label) in enumerate(zip(obs[:3], labels)):
            x = 32 + index * 330
            d.text((x + 8, 150), label, font=font(15, True), fill=MUTED)
            paste_fit(canvas, path, (x, 174, 320, 420))
        if len(obs) >= 9:
            for index, path in enumerate(obs[3:9]):
                x = 32 + (index % 3) * 330
                y = 610 + (index // 3) * 170
                paste_fit(canvas, path, (x, y, 320, 160))
    else:
        paste_fit(canvas, obs[0], (34, 165, 994, 782))
    d.rounded_rectangle((1062, 111, 1580, 383), 14, fill=PANEL)
    source = 'INITIAL STATE' if record is None else 'PRESET PERTURBATION' if record['source'] == 'injected_error' else 'MODEL ACTION'
    d.text((1080, 128), source, font=font(19, True), fill=color)
    d.text((1080, 161), action_label(record), font=font(26, True), fill=INK)
    y = 207
    details = ['Model outputs STOP.', 'Pose stays unchanged.', 'Show final pose in source scene.'] if presentation_final_scene else action_details(record)
    for line in details:
        for wrapped in textwrap.wrap(line, 41):
            d.text((1080, y), wrapped, font=font(19), fill=MUTED)
            y += 26
    d.rounded_rectangle((1062, 398, 1580, 542), 14, fill=PANEL)
    d.text((1080, 412), 'ERROR AT THIS STATE', font=font(18, True), fill=MUTED)
    d.text((1080, 446), f"Rotation  {metrics['rotation_error_deg']:.4f} deg", font=font(23, True), fill=INK)
    d.text((1080, 481), f"ADD-SB / diameter  {metrics['add_sb_fraction']:.5f}", font=font(21, True), fill=INK)
    d.text((1080, 515), 'Success: STOP + rotation < 5 deg + ADD/d < 0.05', font=font(15), fill=MUTED)
    d.text((1080, 566), 'COMPLETE ACTION SEQUENCE', font=font(18, True), fill=MUTED)
    for j in range(n + 1):
        r = records[j - 1] if j else None
        line_color = ORANGE if r and r['source'] == 'injected_error' else GREEN if j == n else BLUE
        y = 600 + j * 43
        if j == index:
            d.rounded_rectangle((1066, y - 5, 1578, y + 34), 7, fill='#31445e')
        d.ellipse((1081, y + 7, 1093, y + 19), fill=line_color if j <= index else '#4f5c6d')
        d.text((1110, y), f'{j:02d}   {action_label(r)}', font=font(20, j == index), fill=INK if j <= index else MUTED)
    footer = (
        'Presentation render only: source scene at unchanged STOP pose; no model action added.'
        if presentation_final_scene else
        'Original evaluation renders; held frames, no interpolated actions. Orange = preset perturbation. STOP repeats the last view.'
    )
    d.text((26, 977), footer, font=font(15), fill=MUTED)
    return canvas


def encode_video(path, frames, durations, fps=10):
    with av.open(str(path), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = SIZE
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium', 'tune': 'stillimage'}
        stream.codec_context.thread_count = 4
        pts = 0
        for image, duration in zip(frames, durations):
            pixels = np.asarray(image)
            for _ in range(round(duration * fps)):
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts = pts
                pts += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == sum(round(x * fps) for x in durations), (path, decoded)
        assert (stream.width, stream.height) == SIZE
    return {'codec': 'h264', 'pixel_format': 'yuv420p', 'width': SIZE[0], 'height': SIZE[1], 'fps': fps, 'frames': decoded, 'duration_seconds': sum(durations)}


def write_html(output, summaries):
    cards = []
    for s in summaries:
        tid = s['trajectory_id']
        buttons = ''.join(f'<button onclick="showFrame(\'{tid}\',{i})">{i:02d} {html.escape(title)}</button>' for i, title in enumerate(s['state_labels_zh']))
        final_scene_link = ''
        if s.get('final_scene_presentation'):
            buttons += f'<button onclick="showFinalScene(\'{tid}\')">成功后原场景图</button>'
            final_scene_link = f' · <a href="{tid}/frames/final_scene.png" target="_blank">成功后原场景图</a>'
        cards.append(f'''<article><h2>{NAMES[s['context_id']]} · {tid}</h2>
<p>监督动作全部匹配 · 执行成功 · 最终旋转误差 {s['final_metrics']['rotation_error_deg']:.4f}° · ADD-SB/直径 {s['final_metrics']['add_sb_fraction']:.5f}</p>
<video controls preload="metadata" poster="{tid}/frames/state_00.png" src="{tid}.mp4"></video>
<p><a href="{tid}.mp4" download>下载 MP4</a> · <a href="{tid}/contact_sheet.png" target="_blank">完整逐步拼图</a>{final_scene_link} · <a href="{tid}/trace.json">动作和状态记录</a></p>
<details><summary>逐步查看图片</summary><nav>{buttons}</nav><img id="{tid}" src="{tid}/frames/state_00.png"></details></article>''')
    output.joinpath('index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lucida · 50 条轨迹 Overfit 展示</title><style>body{max-width:1280px;margin:40px auto;padding:0 24px;background:#101827;color:#edf3fb;font:17px/1.7 system-ui,sans-serif}h1{font-size:32px}h2{font-size:24px}p{color:#abbcd0}a{color:#70b5ff}article{background:#1b2638;padding:24px;border-radius:16px;margin:28px 0}video,img{width:100%;height:auto;border-radius:10px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}button{padding:10px;color:white;background:#31445e;border:0;border-radius:6px;cursor:pointer}summary{cursor:pointer}code{color:#ffbe68}</style>
<h1>Lucida / GizmoAct · 当前 Overfit 成功案例</h1>
<p>检查点：2000 步。展示固定 50 条训练轨迹中的成功案例。每个视频从初始状态开始，展示模型评测时的逐步观测和动作，直到模型输出停止。</p>
<p>蓝色表示模型动作，橙色 <code>Preset perturbation</code> 表示原轨迹预设的扰动，绿色表示停止成功。扰动也保留在完整序列中。切换到六轴视图时，物体局部视角会随之变化。视频只按顺序停留展示离散状态，没有生成中间动作。</p>
<p>动作过程保留模型评测时收到的观测：其中 mesh 以纯橙色诊断材质绘制，所以不显示资产纹理。每个视频在 STOP 后都会用 GLB 自带材质和贴图、原始场景视角再次显示同一成功姿态；这只是展示渲染，不是额外的模型动作。黄框是 gizmo 边界，终点渲染已隐藏该边界，避免把它误看成物体本身。</p>
<p>画布放大不会增加原图细节；front_02 的笔仍受原始小目标分辨率限制。</p>
''' + '\n'.join(cards) + '''<script>function showFrame(id,i){document.getElementById(id).src=id+'/frames/state_'+String(i).padStart(2,'0')+'.png';}function showFinalScene(id){document.getElementById(id).src=id+'/frames/final_scene.png';}</script></html>''', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--step', type=int, default=2000)
    ap.add_argument('--trajectory-ids', nargs='+', default=DEFAULT_IDS)
    args = ap.parse_args()
    dataset = args.dataset_root.resolve()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / 'manifest.json'
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
    contexts = {x.context_id: x for x in manifest.contexts}
    trajectories = {x.trajectory_id: x for x in manifest.trajectories}
    eval_dir = run / f'eval-{args.step}'
    payload = json.loads((eval_dir / 'closed_loop.json').read_text())
    rollouts = {x['trajectory_id']: x for x in payload['rollouts']}
    with (run / f'teacher-forced-{args.step}' / 'per_trajectory_exact.csv').open() as f:
        exact = {x['trajectory_id']: x for x in csv.DictReader(f)}
    summaries = []
    overview_images = []
    for tid in args.trajectory_ids:
        rollout, trajectory = rollouts[tid], trajectories[tid]
        context = contexts[trajectory.context_id]
        records = rollout['actions']
        assert int(exact[tid]['all_actions_exact']) == 1, f'{tid}: incomplete teacher-forced fit'
        assert full_loop_success(rollout['metrics']) == 1, f'{tid}: rollout failed'
        assert rollout['termination'] == 'stop' and isinstance(parse_action(records[-1]['action']), Stop)
        assert len(records) == len(trajectory.turns), f'{tid}: action schedule differs from frozen labels'
        for record, turn in zip(records, trajectory.turns):
            expected_source = 'injected_error' if turn.injected_error else 'model'
            assert record['source'] == expected_source
            assert action_to_text(parse_action(record['action'])) == action_to_text(parse_action(turn.action))
        dirs = list(eval_dir.glob(f'shard_*/rollouts/{tid}'))
        assert len(dirs) == 1, (tid, dirs)
        source_dir = dirs[0]
        case_dir = output / tid
        (case_dir / 'frames').mkdir(parents=True, exist_ok=True)
        (case_dir / 'observations').mkdir(exist_ok=True)
        current = GizmoState(pose(trajectory.initial_pose))
        states = [current]
        for record in records:
            current = execute_action(current, parse_action(record['action']))
            states.append(current)
        target = pose(trajectory.target_pose)
        mesh = trimesh.load(dataset / context.mesh_path, force='mesh', process=False)
        points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=20260922)
        target_points = transform_normalized_points(points, target)
        diameter = object_diameter(target_points)
        metrics = []
        for state in states:
            distance = add_sb(transform_normalized_points(points, state.pose), target_points)
            metrics.append({'add_sb_m': distance, 'add_sb_fraction': distance / diameter, 'rotation_error_deg': rotation_geodesic_deg(state.pose, target)})
        for key, value in metrics[-1].items():
            assert math.isclose(value, float(rollout['metrics'][key]), rel_tol=1e-8, abs_tol=1e-8), (tid, key, value, rollout['metrics'][key])
        frames, trace = [], []
        for index, state in enumerate(states):
            obs_index = min(index, len(records) - 1)
            paths = observation_paths(source_dir, obs_index)
            sources = []
            for path in paths:
                copied = case_dir / 'observations' / path.name
                if not copied.exists():
                    shutil.copy2(path, copied)
                sources.append({'original_path': str(path), 'copied_path': str(copied.relative_to(output)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            frame = draw_frame(tid, context, index, records, metrics[index], paths, state.observation_mode.value, args.step)
            frame_path = case_dir / 'frames' / f'state_{index:02d}.png'
            frame.save(frame_path)
            frames.append(frame)
            trace.append({'state_index': index, 'after_action': records[index - 1] if index else None, 'observation_index': obs_index, 'stop_reuses_last_observation': index == len(records), 'observation_mode': state.observation_mode.value, 'pose': pose_record(state.pose), 'metrics': metrics[index], 'source_images': sources, 'frame': str(frame_path.relative_to(output))})
        # Render every final pose from the source scene camera with the GLB's own
        # material, even when the model finished in its normal scene view.
        final_scene_presentation = True
        presentation = None
        if final_scene_presentation:
            assert np.array_equal(states[-1].pose.position, states[-2].pose.position)
            assert np.array_equal(states[-1].pose.rotation, states[-2].pose.rotation)
            assert np.array_equal(states[-1].pose.size, states[-2].pose.size)
            from lucida_mini.render import render_native_focus_observation
            render_base = case_dir / 'observations' / 'terminal_scene_render.png'
            rendered = render_native_focus_observation(
                dataset / 'contexts' / trajectory.context_id,
                states[-1].pose,
                render_base,
                mode='scene',
                use_source_material=True,
            )
            final_paths = []
            suffixes = ['raw', 'overlay', 'pointcloud'] if len(rendered) == 3 else [f'view_{i:02d}' for i in range(len(rendered))]
            for source_path, suffix in zip(rendered, suffixes):
                final_path = case_dir / 'observations' / f'final_scene_{suffix}.png'
                if source_path.resolve() != final_path.resolve():
                    shutil.copy2(source_path, final_path)
                final_paths.append(final_path)
            final_frame = draw_frame(
                tid, context, len(records), records, metrics[-1], final_paths,
                'scene', args.step, presentation_final_scene=True,
            )
            frame_path = case_dir / 'frames' / 'final_scene.png'
            final_frame.save(frame_path)
            frames.append(final_frame)
            presentation = {
                'type': 'post_stop_presentation_render',
                'is_model_action': False,
                'reason': 'Show final successful pose in the original scene view using the GLB source material.',
                'pose_unchanged_from_stopped_state': True,
                'observation_mode': 'scene',
                'pose': pose_record(states[-1].pose),
                'metrics': metrics[-1],
                'source_images': [
                    {'path': str(path.relative_to(output)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in final_paths
                ],
                'frame': str(frame_path.relative_to(output)),
            }
        durations = [2.5] * (len(frames) - (2 if final_scene_presentation else 1))
        if final_scene_presentation:
            durations.extend([1.5, 4.0])
        else:
            durations.append(3.5)
        video_meta = encode_video(output / f'{tid}.mp4', frames, durations)
        previews = [x.resize((800, 500), Image.Resampling.LANCZOS) for x in frames]
        previews[0].save(output / f'{tid}.gif', save_all=True, append_images=previews[1:], duration=[int(x * 1000) for x in durations], loop=0, optimize=False)
        sheet = Image.new('RGB', (1600, 60 + 500 * math.ceil(len(frames) / 2)), BG)
        sheet_title = f'{tid} | Initial state -> every action -> STOP / SUCCESS' + (' -> SOURCE SCENE VIEW' if final_scene_presentation else '')
        ImageDraw.Draw(sheet).text((24, 15), sheet_title, font=font(23, True), fill=INK)
        for i, preview in enumerate(previews):
            sheet.paste(preview, ((i % 2) * 800, 60 + (i // 2) * 500))
        sheet.save(case_dir / 'contact_sheet.png')
        overview_images.append((tid, previews[0], previews[-1]))
        labels = ['初始状态']
        zh_actions = {'Update pose': '调整姿态', 'Switch view': '切换视角', 'Permute axes': '轴置换', 'Stop / success': '停止成功', 'Preset perturbation': '预设扰动'}
        labels.extend(zh_actions[action_label(r)] for r in records)
        summary = {'trajectory_id': tid, 'context_id': trajectory.context_id, 'checkpoint_step': args.step, 'teacher_forced_all_actions_exact': True, 'rollout_actions_match_frozen_labels': True, 'model_actions': int(rollout['metrics']['actions']), 'preset_perturbations': int(rollout['metrics']['injected_actions']), 'states_including_initial_and_stop': len(states), 'final_scene_presentation': final_scene_presentation, 'final_metrics': rollout['metrics'], 'video': f'{tid}.mp4', 'video_metadata': video_meta, 'durations_seconds': durations, 'state_labels_zh': labels}
        (case_dir / 'trace.json').write_text(json.dumps({'summary': summary, 'protocol': payload['protocol'], 'states': trace, 'post_stop_presentation_render': presentation}, indent=2, ensure_ascii=False) + '\n')
        summaries.append(summary)
        ending = ' + textured source-scene success view' if final_scene_presentation else ''
        print(f'EXPORTED {tid}: {len(states)} states{ending}, {sum(durations):.1f}s, H.264 fully decoded, final metrics verified', flush=True)
    overview = Image.new('RGB', (1600, 72 + 538 * len(overview_images)), BG)
    od = ImageDraw.Draw(overview)
    od.text((24, 18), 'LUCIDA OVERFIT / CHECKPOINT 2000        INITIAL STATE  ->  FINAL SUCCESS', font=font(26, True), fill=INK)
    for i, (tid, initial, final) in enumerate(overview_images):
        y = 72 + i * 538
        od.text((24, y), tid, font=font(22, True), fill=BLUE)
        overview.paste(initial, (0, y + 34))
        overview.paste(final, (800, y + 34))
    overview.save(output / 'overview.png')
    (output / 'index.json').write_text(json.dumps({'manifest': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'evaluation': str(eval_dir), 'protocol': payload['protocol'], 'cases': summaries}, indent=2, ensure_ascii=False) + '\n')
    write_html(output, summaries)
    (output / 'README.md').write_text('# Lucida 当前 Overfit 视频\n\n打开 `index.html` 可观看全部视频及逐步图片。每个 MP4 使用 H.264/yuv420p，1600×1000，10 fps。动作步骤保留模型评测时收到的橙色 mesh 诊断渲染。STOP 后追加使用 GLB 自带材质与贴图的原场景终点画面，物体姿态和成功指标与 STOP 状态一致，也没有增加模型动作；终点隐藏黄色 gizmo 边界框以凸显网格外观。\n\n每条轨迹均已验证：teacher-forced 全部监督动作匹配、执行成功、全部动作与固定轨迹标签一致、回放后的最终几何误差与评测记录一致，且视频全帧可解码。`trace.json` 记录每一步状态、完整动作、误差、图片来源和 SHA256。\n\n图片保存在各案例 `observations/` 与 `frames/`。画布放大不会增加原图细节。\n', encoding='utf-8')
    print(f'Gallery: {output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
