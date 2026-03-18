#!/usr/bin/env python3
"""
Scene Graph Merging Script - Merge scene graphs from multiple frames into a global scene graph.

Merging rules:
1. For the same instance_id appearing with different categories in different frames, choose the most frequent category
2. For relation and parent, choose the combination that most frequently appears with the final category
3. Output global tree structure

Usage:
    python merge_scene_graphs.py \
        --scene_graphs_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/scene_graphs \
        --output_path output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/global_scene_graph.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_all_scene_graphs(scene_graphs_dir: str) -> List[Dict[str, Any]]:
    """Load all scene graph JSON files from the directory."""
    scene_graphs = []
    
    for filename in sorted(os.listdir(scene_graphs_dir)):
        if filename.endswith('.json') and filename != 'all_scene_graphs.json' and filename != 'global_scene_graph.json':
            filepath = os.path.join(scene_graphs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_filename'] = filename
                    scene_graphs.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
    
    print(f"Loaded {len(scene_graphs)} scene graphs")
    return scene_graphs


def collect_instance_info(scene_graphs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Collect all information for each instance_id.
    
    Returns:
        {
            instance_id: {
                'categories': Counter({'chair': 5, 'stool': 2}),
                'relations': Counter({('support', 1): 6, ('attached', 2): 1}),
                'category_relation': defaultdict(Counter),  # category -> Counter of (relation, parent)
                'frames': [list of frame names where this instance appears]
            }
        }
    """
    instance_info = defaultdict(lambda: {
        'categories': Counter(),
        'relations': Counter(),
        'category_relation': defaultdict(Counter),
        'frames': []
    })
    
    for sg_data in scene_graphs:
        frame_name = sg_data.get('frame', 'unknown')
        scene_graph = sg_data.get('scene_graph', {})
        nodes = scene_graph.get('nodes', [])
        edges = scene_graph.get('edges', [])
        
        # Build id -> node mapping
        node_by_id = {n['id']: n for n in nodes}
        
        # Build target_id -> edge mapping
        edge_by_target = {e['target']: e for e in edges}
        
        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type', 'object')
            
            # Skip root nodes (floor, wall)
            if node_type == 'root' or node_id in [1, 2]:
                continue
            
            category = node.get('label', 'unknown')
            instance_id = node.get('instance_id', node_id - 3)  # display_id - 3
            
            # Get edge info for this node
            edge = edge_by_target.get(node_id)
            if edge:
                relation = edge.get('relation', 'support')
                parent = edge.get('source', 1)
            else:
                relation = 'support'
                parent = 1
            
            # Record information
            info = instance_info[instance_id]
            info['categories'][category] += 1
            info['relations'][(relation, parent)] += 1
            info['category_relation'][category][(relation, parent)] += 1
            info['frames'].append(frame_name)
    
    return dict(instance_info)


def merge_instances(instance_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge instance information, selecting the best category and relation for each instance_id.
    
    Rules:
    1. Choose the most frequent category
    2. For that category, choose the most frequently co-occurring (relation, parent)
    """
    merged_objects = []
    
    for instance_id, info in sorted(instance_info.items()):
        # 1. Choose the most frequent category
        if not info['categories']:
            continue
        
        best_category, category_count = info['categories'].most_common(1)[0]
        
        # 2. For that category, choose the most frequently co-occurring (relation, parent)
        category_relations = info['category_relation'][best_category]
        if category_relations:
            (best_relation, best_parent), relation_count = category_relations.most_common(1)[0]
        else:
            # fallback: choose from all relations
            if info['relations']:
                (best_relation, best_parent), relation_count = info['relations'].most_common(1)[0]
            else:
                best_relation, best_parent = 'support', 1
                relation_count = 0
        
        # Statistics
        total_appearances = sum(info['categories'].values())
        all_categories = dict(info['categories'])
        
        merged_objects.append({
            'instance_id': instance_id,
            'display_id': instance_id + 3,
            'category': best_category,
            'relation': best_relation,
            'parent': best_parent,
            'confidence': category_count / total_appearances if total_appearances > 0 else 0,
            'appearances': total_appearances,
            'all_categories': all_categories,
            'num_frames': len(set(info['frames']))
        })
    
    return merged_objects


def build_global_tree(merged_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build global scene graph tree.
    """
    # Root nodes
    nodes = [
        {'id': 1, 'label': 'floor', 'type': 'root'},
        {'id': 2, 'label': 'wall', 'type': 'root'}
    ]
    edges = []
    
    for obj in merged_objects:
        display_id = obj['display_id']
        
        # Add node
        nodes.append({
            'id': display_id,
            'label': obj['category'],
            'type': 'object',
            'instance_id': obj['instance_id'],
            'confidence': obj['confidence'],
            'appearances': obj['appearances'],
            'all_categories': obj['all_categories']
        })
        
        # Add edge
        edges.append({
            'source': obj['parent'],
            'target': display_id,
            'relation': obj['relation']
        })
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def visualize_global_tree(scene_graph: Dict[str, Any]) -> str:
    """Generate text visualization of global scene graph."""
    lines = []
    lines.append("=" * 70)
    lines.append("GLOBAL SCENE GRAPH (Merged from all frames)")
    lines.append("=" * 70)
    
    nodes = scene_graph.get('nodes', [])
    edges = scene_graph.get('edges', [])
    
    # Build tree structure
    nodes_by_id = {n['id']: n for n in nodes}
    children = defaultdict(list)
    
    for edge in edges:
        source = edge['source']
        target = edge['target']
        relation = edge['relation']
        children[source].append((target, relation))
    
    def format_node(node):
        """Format node display."""
        if node['type'] == 'root':
            return f"🏠 [{node['id']}] {node['label']}"
        else:
            conf = node.get('confidence', 0)
            apps = node.get('appearances', 0)
            all_cats = node.get('all_categories', {})
            
            # Show multiple category candidates if any
            if len(all_cats) > 1:
                cats_str = ", ".join([f"{cat}:{cnt}" for cat, cnt in sorted(all_cats.items(), key=lambda x: -x[1])])
                extra = f" (votes: {cats_str})"
            else:
                extra = ""
            
            return f"📦 [{node['id']}] {node['label']} (conf:{conf:.1%}, seen:{apps}x){extra}"
    
    def print_tree(node_id: int, indent: int = 0, relation: str = ""):
        node = nodes_by_id.get(node_id)
        if not node:
            return
        
        # Build indentation
        if indent == 0:
            prefix = ""
        else:
            prefix = "│   " * (indent - 1) + "├── "
        
        rel_str = f"[{relation}] " if relation else ""
        lines.append(f"{prefix}{rel_str}{format_node(node)}")
        
        # Sort child nodes by relation and id
        child_list = sorted(children.get(node_id, []), key=lambda x: (x[1], x[0]))
        for child_id, child_rel in child_list:
            print_tree(child_id, indent + 1, child_rel)
    
    # Print FLOOR tree
    lines.append("\n🔽 FLOOR TREE (supported objects)")
    lines.append("-" * 50)
    print_tree(1)
    
    # Print WALL tree
    lines.append("\n🔽 WALL TREE (attached objects)")
    lines.append("-" * 50)
    print_tree(2)
    
    # Statistics
    total_objects = len([n for n in nodes if n['type'] == 'object'])
    support_edges = len([e for e in edges if e['relation'] == 'support'])
    attached_edges = len([e for e in edges if e['relation'] == 'attached'])
    
    lines.append("\n" + "-" * 50)
    lines.append(f"Total: {total_objects} objects, {support_edges} support, {attached_edges} attached")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_html_visualization(scene_graph: Dict[str, Any], output_path: str):
    """Generate HTML visualization."""
    nodes_js = json.dumps(scene_graph['nodes'])
    edges_js = json.dumps(scene_graph['edges'])
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Global Scene Graph</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .container {{ display: flex; gap: 30px; }}
        .tree-container {{ flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tree-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #2196F3; }}
        .node {{ margin: 4px 0; }}
        .node-content {{ 
            display: inline-block; 
            padding: 6px 12px; 
            border-radius: 6px; 
            font-size: 13px;
        }}
        .root {{ background: #4CAF50; color: white; }}
        .object {{ background: #2196F3; color: white; }}
        .relation {{ color: #FF9800; font-weight: bold; font-size: 12px; margin-right: 5px; }}
        .tree-level {{ margin-left: 25px; border-left: 2px solid #ddd; padding-left: 15px; }}
        .confidence {{ font-size: 11px; opacity: 0.9; }}
        .votes {{ font-size: 10px; color: #FFE082; margin-top: 2px; }}
        .stats {{ margin-top: 20px; padding: 15px; background: #fff; border-radius: 8px; }}
        pre {{ background: #263238; color: #ECEFF1; padding: 15px; border-radius: 8px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>🌳 Global Scene Graph</h1>
    <p>Merged from all frame-level scene graphs using majority voting.</p>
    
    <div class="container">
        <div class="tree-container">
            <div class="tree-title">🔽 FLOOR Tree (supported)</div>
            <div id="floor-tree"></div>
        </div>
        <div class="tree-container">
            <div class="tree-title">🔽 WALL Tree (attached)</div>
            <div id="wall-tree"></div>
        </div>
    </div>
    
    <div class="stats" id="stats"></div>
    
    <h2>Raw Data</h2>
    <pre id="json-data"></pre>
    
    <script>
        const nodes = {nodes_js};
        const edges = {edges_js};
        
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
            nodeDiv.className = 'node';
            
            let html = '';
            if (relation) {{
                html += `<span class="relation">[${{relation}}]</span>`;
            }}
            
            const nodeClass = node.type === 'root' ? 'root' : 'object';
            html += `<span class="node-content ${{nodeClass}}">`;
            html += `[${{node.id}}] ${{node.label}}`;
            
            if (node.type === 'object') {{
                const conf = (node.confidence * 100).toFixed(0);
                html += `<span class="confidence"> (${{conf}}%, ${{node.appearances}}x)</span>`;
                
                if (node.all_categories && Object.keys(node.all_categories).length > 1) {{
                    const votes = Object.entries(node.all_categories)
                        .sort((a,b) => b[1] - a[1])
                        .map(([cat, cnt]) => `${{cat}}:${{cnt}}`)
                        .join(', ');
                    html += `<div class="votes">votes: ${{votes}}</div>`;
                }}
            }}
            html += '</span>';
            
            nodeDiv.innerHTML = html;
            container.appendChild(nodeDiv);
            
            const nodeChildren = children[nodeId] || [];
            if (nodeChildren.length > 0) {{
                const childContainer = document.createElement('div');
                childContainer.className = 'tree-level';
                nodeChildren.sort((a,b) => a.target - b.target);
                nodeChildren.forEach(c => renderTree(c.target, childContainer, c.relation));
                container.appendChild(childContainer);
            }}
        }}
        
        renderTree(1, document.getElementById('floor-tree'));
        renderTree(2, document.getElementById('wall-tree'));
        
        // Stats
        const totalObjects = nodes.filter(n => n.type === 'object').length;
        const supportEdges = edges.filter(e => e.relation === 'support').length;
        const attachedEdges = edges.filter(e => e.relation === 'attached').length;
        document.getElementById('stats').innerHTML = 
            `<strong>Statistics:</strong> ${{totalObjects}} objects, ${{supportEdges}} support relations, ${{attachedEdges}} attached relations`;
        
        document.getElementById('json-data').textContent = JSON.stringify({{nodes, edges}}, null, 2);
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Merge multi-frame scene graphs into global scene graph")
    parser.add_argument(
        "--scene_graphs_dir",
        type=str,
        required=True,
        help="Scene graphs directory (containing multiple frame.json files)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path (default: <scene_graphs_dir>/global_scene_graph.json)",
    )
    
    args = parser.parse_args()
    
    scene_graphs_dir = args.scene_graphs_dir
    if not os.path.exists(scene_graphs_dir):
        raise FileNotFoundError(f"Scene graphs directory not found: {scene_graphs_dir}")
    
    output_path = args.output_path or os.path.join(scene_graphs_dir, "global_scene_graph.json")
    output_dir = os.path.dirname(output_path)
    
    # 1. Load all scene graphs
    print("Loading scene graphs...")
    scene_graphs = load_all_scene_graphs(scene_graphs_dir)
    
    if not scene_graphs:
        print("No scene graphs found!")
        return
    
    # 2. Collect information for each instance
    print("Collecting instance information...")
    instance_info = collect_instance_info(scene_graphs)
    print(f"Found {len(instance_info)} unique instances")
    
    # 3. Merge instances
    print("Merging instances...")
    merged_objects = merge_instances(instance_info)
    print(f"Merged into {len(merged_objects)} objects")
    
    # Print merge details
    print("\n--- Merge Details ---")
    for obj in merged_objects:
        cats = obj['all_categories']
        if len(cats) > 1:
            cats_str = ", ".join([f"{cat}:{cnt}" for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])])
            print(f"  Instance {obj['instance_id']} (ID {obj['display_id']}): {obj['category']} (from: {cats_str})")
        else:
            print(f"  Instance {obj['instance_id']} (ID {obj['display_id']}): {obj['category']}")
    print("---\n")
    
    # 4. Build global tree
    print("Building global tree...")
    global_scene_graph = build_global_tree(merged_objects)
    
    # 5. Visualization
    text_viz = visualize_global_tree(global_scene_graph)
    print(text_viz)
    
    # 6. Save results
    # JSON
    result = {
        'total_frames': len(scene_graphs),
        'total_instances': len(merged_objects),
        'scene_graph': global_scene_graph,
        'merged_objects': merged_objects,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {output_path}")
    
    # HTML
    html_path = output_path.replace('.json', '.html')
    generate_html_visualization(global_scene_graph, html_path)
    print(f"Saved HTML: {html_path}")
    
    # TXT
    txt_path = output_path.replace('.json', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_viz)
    print(f"Saved TXT: {txt_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
