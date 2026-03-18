#!/usr/bin/env python3
"""
Scene Graph Inference Script - Use Qwen3-VL to infer scene graphs from images with ID labels.

Features:
1. Read id_scene.png (scene image with numbered labels)
2. Identify object category for each ID
3. Infer support/attachment relations for each object (supports multi-level nesting)
4. Generate scene graph and save as JSON + visualization

Usage:
    # Single frame inference
    python infer_scene_graph.py \
        --id_scene_path output/.../instance_project/120/id_scene.png
    
    # Batch process entire scene
    python infer_scene_graph.py \
        --instance_project_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/instance_project \
        --output_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/scene_graphs
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path: str = "Qwen/Qwen3-VL-8B-Thinking"):
    """Load Qwen3-VL model."""
    print(f"Loading model: {model_path} ...")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # GLIBC too old for flash_attn, use sdpa instead
        trust_remote_code=True,
    ).cuda()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    print("Model loaded successfully.")
    return model, processor


def extract_ids_from_image_name(image_path: str) -> List[int]:
    """
    Extract all visible instance IDs from the instance_project folder for this frame.
    """
    frame_dir = os.path.dirname(image_path)
    ids = []
    
    for fname in os.listdir(frame_dir):
        if fname.startswith("instance_") and fname.endswith(".png") and "_mask" not in fname:
            # instance_4_desk.png -> 4
            match = re.match(r"instance_(\d+)_", fname)
            if match:
                ids.append(int(match.group(1)))
    
    return sorted(ids)


def infer_objects_and_relations(
    model,
    processor,
    image_path: str,
    object_ids: List[int],
) -> Dict[str, Any]:
    """
    Use Qwen3-VL to infer object categories and support relations for each ID in the image.
    
    Args:
        model: Qwen3-VL model
        processor: Processor
        image_path: Path to id_scene.png
        object_ids: List of object IDs in the image
    
    Returns:
        Inference result dictionary
    """
    # Convert ID to display ID (id + 3)
    display_ids = [oid + 3 for oid in object_ids]
    display_ids_str = ", ".join(map(str, display_ids))
    num_objects = len(display_ids)
    
    prompt = f'''You are analyzing a room scene image with numbered labels (green boxes with red numbers).

**Visible object IDs**: {display_ids_str} (Total: {num_objects} objects)

**Hidden root nodes** (not shown in image):
- ID 1 = "floor" (ground surface)
- ID 2 = "wall" (vertical wall surface)

## STEP-BY-STEP ANALYSIS (Follow this order strictly):

### Step 1: Identify ALL objects
For EACH of the {num_objects} IDs ({display_ids_str}), identify what the object is.
- You MUST identify ALL {num_objects} objects, no skipping!
- IMPORTANT: Two objects with the SAME category (e.g., two chairs) are DIFFERENT objects if they are at different positions!
- Only mark as "duplicate" if two IDs point to the EXACT SAME physical object (overlapping labels)

### Step 2: Build FLOOR tree (support relations)
Think about what objects are SUPPORTED BY FLOOR (standing on ground):
- Furniture: desk, table, chair, sofa, bed, cabinet, wardrobe, shelf, bookcase, dresser, nightstand, couch, bench, stool
- Appliances: refrigerator, washing machine, TV stand
- Other: carpet, rug, plant pot, trash can, box, luggage, suitcase

Then think: what objects are ON TOP of these floor-supported objects?
- Example chain: floor → desk → lamp → cup

### Step 3: Build WALL tree (attached relations)  
Think about what objects are ATTACHED TO WALL (hanging/fixed on vertical surface):
- ONLY these types: picture, painting, photo frame, mirror, clock, poster, whiteboard, TV (wall-mounted), window, door, curtain rod, light switch, outlet
- Curtains are attached to wall (via curtain rod)

Then think: what objects might be ON these wall-attached objects?
- Example: wall → shelf → books

## PHYSICAL COMMON SENSE RULES:
1. **Cabinets, wardrobes, bookshelves** → ALWAYS supported by floor (parent=1), NOT attached to wall!
2. **Tables, desks, chairs, beds** → ALWAYS supported by floor (parent=1)
3. **Wall-attached** is RARE, only for: pictures, mirrors, clocks, posters, wall-mounted TVs, curtains, windows, doors
4. **Curtains** → attached to wall (parent=2)
5. **Objects on furniture** → parent is that furniture's ID, relation is "support"
6. **Small items on desk** (lamp, monitor, keyboard, cup) → supported by desk

## OUTPUT FORMAT:
```json
{{
  "objects": [
    {{"id": <display_id>, "category": "<name>", "relation": "support|attached", "parent": <parent_id>}}
  ]
}}
```

## CRITICAL REQUIREMENTS:
- You MUST output exactly {num_objects} objects (one for each ID: {display_ids_str})
- Do NOT skip any object!
- If an object looks like "floor" or "wall" itself, still include it but note category as "floor_area" or "wall_section"
- If unsure, default to: relation="support", parent=1 (floor)

Now analyze the image and output the complete JSON with all {num_objects} objects.
'''

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32768)  # Increase length limit to support multi-object output
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def post_process_objects(objects: List[Dict], expected_ids: List[int]) -> List[Dict]:
    """
    Post-processing: Filter and correct physical errors.
    
    1. Filter out objects with categories close to floor/wall
    2. Correct common physical errors (e.g., cabinets should not be attached to wall)
    
    Note: Do NOT deduplicate based on category! Two chairs are two different objects and should both be kept.
    Only deduplicate when VLM explicitly marks as "duplicate" or "same_as_XX".
    """
    # Categories to filter (are themselves floor/wall)
    FILTER_CATEGORIES = {
        'floor', 'wall', 'ground', 'ceiling', 'floor_area', 'wall_section',
        'flooring', 'wall_surface', 'room', 'space'
    }
    
    # Categories that must be floor-supported (cannot be attached to wall)
    MUST_BE_FLOOR_SUPPORTED = {
        'cabinet', 'wardrobe', 'closet', 'dresser', 'bookshelf', 'bookcase',
        'shelf', 'shelving', 'desk', 'table', 'chair', 'sofa', 'couch',
        'bed', 'mattress', 'nightstand', 'bench', 'stool', 'armchair',
        'refrigerator', 'fridge', 'washing_machine', 'dryer', 'oven', 'stove',
        'tv_stand', 'entertainment_center', 'sideboard', 'buffet',
        'filing_cabinet', 'storage_cabinet', 'kitchen_cabinet',
        'plant', 'potted_plant', 'trash_can', 'garbage_can', 'bin',
        'box', 'luggage', 'suitcase', 'bag', 'backpack',
        'carpet', 'rug', 'mat', 'ottoman', 'footstool'
    }
    
    # Categories that can only be wall-attached
    MUST_BE_WALL_ATTACHED = {
        'picture', 'painting', 'photo', 'photo_frame', 'frame',
        'mirror', 'clock', 'wall_clock', 'poster', 'art',
        'whiteboard', 'blackboard', 'bulletin_board',
        'tv', 'television', 'monitor',  # wall-mounted TV
        'light_switch', 'outlet', 'socket',
        'curtain', 'drape', 'blind', 'shade',
        'window', 'door', 'vent', 'air_vent'
    }
    
    filtered_objects = []
    
    for obj in objects:
        obj_id = obj.get('id')
        category = obj.get('category', '').lower().replace(' ', '_')
        relation = obj.get('relation', 'support')
        parent = obj.get('parent', 1)
        
        # 1. Filter out objects with categories close to floor/wall
        if category in FILTER_CATEGORIES:
            print(f"  [Filter] Skipping ID {obj_id}: category '{category}' is floor/wall type")
            continue
        
        # 2. Filter out objects marked as duplicate by VLM (e.g., category contains "duplicate" or "same")
        if 'duplicate' in category or 'same_as' in category:
            print(f"  [Filter] Skipping ID {obj_id}: marked as duplicate")
            continue
        
        # 3. Correct physical errors
        category_base = category.split('_')[0] if '_' in category else category
        
        # If object must be floor-supported but marked as attached to wall
        if (category in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED):
            if relation == 'attached' and parent == 2:
                print(f"  [Fix] ID {obj_id} '{category}': changed from wall-attached to floor-supported")
                relation = 'support'
                parent = 1
        
        # If object must be wall-attached but marked as floor-supported with parent=1
        if (category in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED):
            if relation == 'support' and parent == 1:
                # Only correct for objects that should clearly be on wall
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    print(f"  [Fix] ID {obj_id} '{category}': changed from floor-supported to wall-attached")
                    relation = 'attached'
                    parent = 2
        
        filtered_objects.append({
            'id': obj_id,
            'category': obj.get('category', 'unknown'),  # Keep original format
            'relation': relation,
            'parent': parent
        })
    
    return filtered_objects


def parse_model_output(output_text: str) -> Dict[str, Any]:
    """Parse model output JSON (supports Thinking model's </think> tag)."""
    
    # If there's a </think> tag, only take content after it
    if '</think>' in output_text:
        output_text = output_text.split('</think>')[-1].strip()
    
    # Try to extract from ```json``` code block
    code_match = re.search(r'```json\s*([\s\S]*?)\s*```', output_text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to extract the last complete JSON object (containing "objects")
    json_matches = re.findall(r'\{[^{}]*"objects"[^{}]*\[[\s\S]*?\][^{}]*\}', output_text)
    if json_matches:
        for json_str in reversed(json_matches):  # Try from end to beginning
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Try to extract any JSON object
    json_match = re.search(r'\{\s*"objects"\s*:\s*\[[\s\S]*?\]\s*\}', output_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return {"objects": [], "parse_error": True, "raw_output": output_text}


def build_scene_graph(parsed_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build scene graph data structure.
    
    Scene graph structure:
    - nodes: All nodes (including floor, wall and detected objects)
    - edges: All edges (support/attachment relations)
    """
    # Initialize root nodes
    nodes = [
        {"id": 1, "label": "floor", "type": "root"},
        {"id": 2, "label": "wall", "type": "root"},
    ]
    edges = []
    
    objects = parsed_result.get("objects", [])
    
    for obj in objects:
        obj_id = obj.get("id")
        category = obj.get("category", "unknown")
        relation = obj.get("relation", "support")
        parent = obj.get("parent", 1)
        
        # Add node
        nodes.append({
            "id": obj_id,
            "label": category,
            "type": "object",
            "instance_id": obj_id - 3,  # Original instance_id
        })
        
        # Add edge
        edges.append({
            "source": parent,
            "target": obj_id,
            "relation": relation,
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
    }


def visualize_scene_graph_text(scene_graph: Dict[str, Any]) -> str:
    """Generate text visualization of scene graph (supports multi-level nesting)."""
    lines = []
    lines.append("=" * 60)
    lines.append("SCENE GRAPH (Multi-level Tree)")
    lines.append("=" * 60)
    
    # Build tree structure
    nodes_by_id = {n["id"]: n for n in scene_graph["nodes"]}
    children = {n["id"]: [] for n in scene_graph["nodes"]}
    
    for edge in scene_graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        relation = edge["relation"]
        if source in children:
            children[source].append((target, relation))
    
    def print_tree(node_id: int, indent: int = 0, relation: str = "", prefix: str = ""):
        node = nodes_by_id.get(node_id)
        if not node:
            return
        
        # Build tree connector
        if indent == 0:
            connector = ""
        else:
            connector = f"{'│   ' * (indent-1)}├── "
        
        rel_str = f"[{relation}] " if relation else ""
        node_type = "🏠" if node["type"] == "root" else "📦"
        lines.append(f"{connector}{rel_str}{node_type} [{node['id']}] {node['label']}")
        
        child_list = children.get(node_id, [])
        for i, (child_id, child_rel) in enumerate(child_list):
            print_tree(child_id, indent + 1, child_rel)
    
    # Print floor tree
    lines.append("\n🔽 FLOOR TREE (Root: floor, ID=1)")
    lines.append("-" * 40)
    print_tree(1)
    
    # Print wall tree
    lines.append("\n🔽 WALL TREE (Root: wall, ID=2)")
    lines.append("-" * 40)
    print_tree(2)
    
    # Statistics
    lines.append("\n" + "-" * 40)
    total_objects = len([n for n in scene_graph["nodes"] if n["type"] == "object"])
    support_edges = len([e for e in scene_graph["edges"] if e["relation"] == "support"])
    attached_edges = len([e for e in scene_graph["edges"] if e["relation"] == "attached"])
    lines.append(f"Statistics: {total_objects} objects, {support_edges} support relations, {attached_edges} attached relations")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_scene_graph_html(scene_graph: Dict[str, Any], output_path: str):
    """Generate HTML visualization of scene graph (using simple tree layout)."""
    nodes = scene_graph["nodes"]
    edges = scene_graph["edges"]
    
    # Generate JavaScript data for nodes and edges
    nodes_js = json.dumps(nodes)
    edges_js = json.dumps(edges)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Scene Graph Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .tree {{ margin: 20px 0; }}
        .tree-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
        .node {{ 
            display: inline-block; 
            padding: 8px 16px; 
            margin: 4px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .root {{ background: #4CAF50; color: white; }}
        .object {{ background: #2196F3; color: white; }}
        .relation {{ color: #666; font-size: 12px; }}
        .tree-level {{ margin-left: 30px; border-left: 2px solid #ddd; padding-left: 15px; }}
        h1 {{ color: #333; }}
        .legend {{ margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Scene Graph</h1>
    
    <div class="legend">
        <strong>Legend:</strong>
        <span class="node root">Root (floor/wall)</span>
        <span class="node object">Object</span>
        <span class="relation">--[relation]--></span>
    </div>
    
    <div id="graph"></div>
    
    <h2>Raw Data</h2>
    <pre id="json-data"></pre>
    
    <script>
        const nodes = {nodes_js};
        const edges = {edges_js};
        
        // Build adjacency list
        const children = {{}};
        nodes.forEach(n => children[n.id] = []);
        edges.forEach(e => {{
            if (children[e.source]) {{
                children[e.source].push({{target: e.target, relation: e.relation}});
            }}
        }});
        
        const nodesById = {{}};
        nodes.forEach(n => nodesById[n.id] = n);
        
        function renderTree(nodeId, container, relation = '') {{
            const node = nodesById[nodeId];
            if (!node) return;
            
            const nodeDiv = document.createElement('div');
            
            if (relation) {{
                const relSpan = document.createElement('span');
                relSpan.className = 'relation';
                relSpan.textContent = `--[${{relation}}]--> `;
                nodeDiv.appendChild(relSpan);
            }}
            
            const labelSpan = document.createElement('span');
            labelSpan.className = `node ${{node.type}}`;
            labelSpan.textContent = `[${{node.id}}] ${{node.label}}`;
            nodeDiv.appendChild(labelSpan);
            
            container.appendChild(nodeDiv);
            
            const nodeChildren = children[nodeId] || [];
            if (nodeChildren.length > 0) {{
                const childContainer = document.createElement('div');
                childContainer.className = 'tree-level';
                nodeChildren.forEach(c => renderTree(c.target, childContainer, c.relation));
                container.appendChild(childContainer);
            }}
        }}
        
        const graphDiv = document.getElementById('graph');
        
        // Floor tree
        const floorTitle = document.createElement('div');
        floorTitle.className = 'tree-title';
        floorTitle.textContent = 'FLOOR Tree (supported objects):';
        graphDiv.appendChild(floorTitle);
        
        const floorTree = document.createElement('div');
        floorTree.className = 'tree';
        renderTree(1, floorTree);
        graphDiv.appendChild(floorTree);
        
        // Wall tree
        const wallTitle = document.createElement('div');
        wallTitle.className = 'tree-title';
        wallTitle.textContent = 'WALL Tree (attached objects):';
        graphDiv.appendChild(wallTitle);
        
        const wallTree = document.createElement('div');
        wallTree.className = 'tree';
        renderTree(2, wallTree);
        graphDiv.appendChild(wallTree);
        
        // Show raw JSON
        document.getElementById('json-data').textContent = JSON.stringify({{nodes, edges}}, null, 2);
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def process_single_frame(
    model,
    processor,
    id_scene_path: str,
    output_dir: str,
    frame_name: str,
) -> Optional[Dict[str, Any]]:
    """Process single frame and save results."""
    print(f"\n{'='*60}")
    print(f"Processing frame: {frame_name}")
    print(f"Input: {id_scene_path}")
    
    # Extract all instance IDs in this frame
    object_ids = extract_ids_from_image_name(id_scene_path)
    print(f"Detected object IDs (instance_id): {object_ids}")
    print(f"Display IDs (instance_id + 3): {[oid + 3 for oid in object_ids]}")
    
    if not object_ids:
        print("No objects found in this frame. Skipping.")
        return None
    
    # Inference
    print("Running inference...")
    raw_output = infer_objects_and_relations(
        model, processor, id_scene_path, object_ids
    )
    
    print("\n--- Model Raw Output ---")
    print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
    print("--- End of Raw Output ---\n")
    
    # Parse output
    parsed_result = parse_model_output(raw_output)
    
    # Post-processing: filter, deduplicate, correct physical errors
    display_ids = [oid + 3 for oid in object_ids]
    if parsed_result.get("objects"):
        print(f"Before post-processing: {len(parsed_result['objects'])} objects")
        parsed_result["objects"] = post_process_objects(parsed_result["objects"], display_ids)
        print(f"After post-processing: {len(parsed_result['objects'])} objects")
    
    # Build scene graph
    scene_graph = build_scene_graph(parsed_result)
    
    # Generate text visualization
    text_viz = visualize_scene_graph_text(scene_graph)
    print(text_viz)
    
    # Save results
    # 1. JSON format
    json_path = os.path.join(output_dir, f"{frame_name}.json")
    result_data = {
        "frame": frame_name,
        "scene_graph": scene_graph,
        "raw_output": raw_output,
        "parsed_result": parsed_result,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")
    
    # 2. HTML visualization
    html_path = os.path.join(output_dir, f"{frame_name}.html")
    generate_scene_graph_html(scene_graph, html_path)
    print(f"Saved: {html_path}")
    
    # 3. Text format
    txt_path = os.path.join(output_dir, f"{frame_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_viz)
    print(f"Saved: {txt_path}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Scene graph inference")
    parser.add_argument(
        "--id_scene_path",
        type=str,
        default=None,
        help="Single frame mode: path to id_scene.png",
    )
    parser.add_argument(
        "--instance_project_dir",
        type=str,
        default=None,
        help="Batch mode: path to instance_project directory (containing multiple frame subdirectories)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <instance_project_dir>/../scene_graphs)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="Qwen3-VL model name or path",
    )
    
    args = parser.parse_args()
    
    # Determine run mode
    if args.id_scene_path and args.instance_project_dir:
        parser.error("Please specify only one of --id_scene_path or --instance_project_dir")
    
    if not args.id_scene_path and not args.instance_project_dir:
        parser.error("Please specify --id_scene_path (single frame) or --instance_project_dir (batch)")
    
    # Load model (only once)
    model, processor = load_model(args.model_path)
    
    if args.id_scene_path:
        # ========== Single frame mode ==========
        if not os.path.exists(args.id_scene_path):
            raise FileNotFoundError(f"id_scene not found: {args.id_scene_path}")
        
        frame_dir = os.path.dirname(args.id_scene_path)
        frame_name = os.path.basename(frame_dir)
        
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(frame_dir)  # instance_project directory
            output_dir = os.path.join(os.path.dirname(output_dir), "scene_graphs")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output dir: {output_dir}")
        
        process_single_frame(model, processor, args.id_scene_path, output_dir, frame_name)
        
    else:
        # ========== Batch mode ==========
        instance_project_dir = args.instance_project_dir
        if not os.path.exists(instance_project_dir):
            raise FileNotFoundError(f"instance_project_dir not found: {instance_project_dir}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(os.path.dirname(instance_project_dir), "scene_graphs")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output dir: {output_dir}")
        
        # Iterate through all frame directories
        frame_dirs = sorted([
            d for d in os.listdir(instance_project_dir)
            if os.path.isdir(os.path.join(instance_project_dir, d))
        ])
        
        print(f"Found {len(frame_dirs)} frames to process")
        
        all_results = {}
        for frame_name in frame_dirs:
            id_scene_path = os.path.join(instance_project_dir, frame_name, "id_scene.png")
            
            if not os.path.exists(id_scene_path):
                print(f"Warning: id_scene.png not found in {frame_name}, skipping")
                continue
            
            result = process_single_frame(
                model, processor, id_scene_path, output_dir, frame_name
            )
            
            if result:
                all_results[frame_name] = result
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Save summary results
        summary_path = os.path.join(output_dir, "all_scene_graphs.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_frames": len(all_results),
                "frames": list(all_results.keys()),
            }, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_path}")
        print(f"Processed {len(all_results)} frames successfully")
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
