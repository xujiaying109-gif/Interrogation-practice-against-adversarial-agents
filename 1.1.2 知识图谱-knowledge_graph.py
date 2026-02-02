from collections import defaultdict
from typing import List, Dict, Any, Optional
from openai import OpenAI
from schemas import DeceptionOperator
import json

class DualLayerKnowledgeGraph:
    UNKNOWN_MARK = "UNKNOWN_ENTITY"

    # 欺骗算子规则 → 提示指令映射
    DECEPTION_TO_PROMPT_MAP = {
        DeceptionOperator.DISTORT: [
            "将{原关系}重新解释为{新关系}",
            "强调这只是{新解释}，不是{原含义}"
        ],
        DeceptionOperator.OMIT: [
            "避免提及{被删除实体}",
            "当被问到{相关话题}时转移话题"
        ],
        DeceptionOperator.FABRICATE: [
            "可以提及{捏造证据}的存在",
            "用{捏造细节}增加可信度"
        ],
        DeceptionOperator.RATIONALIZE: [
            "将{可疑行为}解释为{正当理由}",
            "强调这是{合理场景}下的正常行为"
        ],
        DeceptionOperator.DENY: [
            "完全否认相关事实的存在",
            "表现出对该信息的极度陌生感"
        ]
    }

    # ==========================
    # 第一部分：初始化
    # ==========================

    def __init__(self, case_data: Dict[str, Any], model_name: str):
        """
        初始化知识图谱，绑定原始剧本字典引用，实现数据同步。
        """
        self.case_data = case_data
        
        # 确保基础结构存在并建立引用
        if "fake_story" not in self.case_data: self.case_data["fake_story"] = []
        if "context_memory" not in self.case_data: self.case_data["context_memory"] = []
        
        self.ground_truth = self.case_data.get("ground_truth", [])
        self.fake_story = self.case_data["fake_story"]
        self.context_memory = self.case_data["context_memory"]

        self.entity_set = self._extract_all_entities()
        self.index = self._build_index()

        self.client = OpenAI(
            api_key="sk-kbkuyxwmxdgijphenfnvgrrsvqorbafgrbtrctxzhzmqeced",
            base_url="http://222.171.219.26:20001/v1/chat/completion"
        )
        self.model_name = model_name #qwen3-30b-a3b-instruct-2507

    # ==========================
    # 第二部分：交互接口
    # ==========================

    def retrieve_by_entities(self, user_input: str, entities: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
            results = {"事实": [], "虚假": [], "记忆": []}
            if not entities:
                return results

            candidate_pool = []
            seen_triples = set() 
            
            layer_map_inv = {"true": "事实", "fake": "虚假", "context": "记忆"}

            for ent in entities:
                hits = self.index.get(ent, [])
                for hit in hits:
                    triple_str = f"[{layer_map_inv[hit['layer']]}] {hit['data'].get('subject')} --{hit['data'].get('predicate')}--> {hit['data'].get('object')} (描述: {hit['data'].get('narrative', '无')})"
                    if triple_str not in seen_triples:
                        candidate_pool.append({
                            "id": len(candidate_pool),
                            "description": triple_str,
                            "original_data": hit['data'],
                            "layer": layer_map_inv[hit['layer']]
                        })
                        seen_triples.add(triple_str)

            if not candidate_pool:
                return results

            #  LLM prompt
            candidates_formatted = "\n".join([f"{c['id']}. {c['description']}" for c in candidate_pool])
            
            system_prompt = (
                "你是一个审讯系统检索专家。你的任务是从候选知识库中挑选出与【用户提问】最相关的知识三元组。\n"
                f"请从以下候选列表中选出最相关的 Top-{top_k} 个，并按相关性降序排列。\n"
                "只返回三元组的 ID 列表，格式为 JSON 数组，例如：[1, 5, 2]。"
            )
            
            user_payload = (
                f"用户提问：'{user_input}'\n\n"
                f"候选三元组列表：\n{candidates_formatted}\n\n"
                "请选出最相关的 ID："
            )

            try:
                #  调用 LLM 进行语义筛选
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload}
                    ],
                    temperature=0,
                    response_format={ "type": "json_object" } 
                )
                
                content = resp.choices[0].message.content.strip()
                if content.startswith("["):
                    selected_ids = json.loads(content)
                else:
                    selected_ids = json.loads(content).get("selected_ids", [])

                for idx in selected_ids:
                    if 0 <= idx < len(candidate_pool):
                        candidate = candidate_pool[idx]
                        results[candidate["layer"]].append(candidate["original_data"])

            except Exception as e:
                print(f"LLM 检索异常: {e}")
                for c in candidate_pool[:top_k]:
                    results[c["layer"]].append(c["original_data"])

            return results

    def add_context_triple(self, triple_data: Dict[str, Any]):
            """接收 CoT 映射出的原始数据，更新嫌疑人的口供记忆。"""
            standard_triple: Triple = {
                "subject": str(triple_data.get("subject", "")),
                "predicate": str(triple_data.get("predicate", "")),
                "object": str(triple_data.get("object", "")),
                "evidence_strength": float(triple_data.get("evidence_strength", 0.3)),
                "narrative": str(triple_data.get("narrative", "")),
                "meta": str(triple_data.get("meta", "")) 
            }

            self.context_memory.append(standard_triple)
            self._update_index_single("context", standard_triple)
            
            if standard_triple["subject"]:
                self.entity_set.add(standard_triple["subject"])
            if standard_triple["object"]:
                self.entity_set.add(standard_triple["object"])

    def generate_fake_response_based_on_truth(self, user_input: str, retrieved_truth: List[Dict]) -> Dict[str, Any]:
        """基于证据强度和剧本事实，应用欺骗算子生成谎言弹药。"""
        if not retrieved_truth:
            return {"fake_triples": [], "deception_type": DeceptionOperator.NONE}
        
        mapping_hint = json.dumps(self.DECEPTION_TO_PROMPT_MAP, ensure_ascii=False)

        system_prompt = (
            "你是一个审讯对抗专家。请将提供的【事实三元组】转化为【谎言三元组】。\n"
            f"遵循以下算子指令规范：{mapping_hint}\n"
            "逻辑准则：\n"
            "1. 证据强度(evidence_strength) > 0.7 时，必须使用 RATIONALIZE 或 DISTORT，严禁直接 DENY。\n"
            "2. 每个谎言必须包含详细的 narrative（叙述），作为嫌疑人的标准口径。\n"
            "3. 必须输出严格 JSON。"
        )

        user_payload = {
            "user_input": user_input,
            "real_facts": retrieved_truth,
            "output_format": {
                "fake_triples": [
                    {
                        "subject": "主体",
                        "predicate": "伪造动作",
                        "object": "伪造客体",
                        "evidence_strength": "float(0-1)",
                        "meta": {"operator": "DeceptionOperator名"},
                        "narrative": "嫌疑人的辩解话术文本"
                    }
                ],
                "primary_operator": "本次主导算子"
            }
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                temperature=0.7
            )
            data = json.loads(resp.choices[0].message.content.strip(" `json\n"))
            generated = data.get("fake_triples", [])

            for ft in generated:
                if ft not in self.fake_story:
                    self.fake_story.append(ft)
                    self._update_index_single("fake", ft)
            
            return {
                "fake_triples": generated,
                "deception_type": data.get("primary_operator", DeceptionOperator.DISTORT)
            }
        except Exception as e:
            print(f"动态欺骗生成异常: {e}")
            return {"fake_triples": [], "deception_type": DeceptionOperator.NONE}

    # ==========================
    # 第三部分：内部索引辅助逻辑
    # ==========================

    def _extract_all_entities(self) -> set:
        """从三层存储中提取所有涉及的实体名。"""
        entities = set()
        for layer in [self.ground_truth, self.fake_story, self.context_memory]:
            for t in layer:
                entities.add(t.get("subject"))
                entities.add(t.get("object"))
        return {e for e in entities if e}

    def _build_index(self) -> defaultdict:
        """根据实体名构建指向三元组的快速索引。"""
        idx = defaultdict(list)
        for t in self.ground_truth: self._update_index_single("true", t, idx)
        for t in self.fake_story: self._update_index_single("fake", t, idx)
        for t in self.context_memory: self._update_index_single("context", t, idx)
        return idx

    def _update_index_single(self, layer_name: str, triple: Dict, index_obj=None):
        """更新单个三元组到索引对象。"""
        if index_obj is None: index_obj = self.index
        s, o = triple.get("subject"), triple.get("object")
        item = {"layer": layer_name, "data": triple}
        if s: index_obj[s].append(item)
        if o: index_obj[o].append(item)

    def clear_context_memory(self):
        """清空对话记忆并重置索引"""
        self.context_memory.clear()
        for entity in list(self.index.keys()):
            self.index[entity] = [i for i in self.index[entity] if i["layer"] != "context"]
            if not self.index[entity]: del self.index[entity]
        self.entity_set = self._extract_all_entities()
