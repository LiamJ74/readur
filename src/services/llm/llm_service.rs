use std::sync::Arc;
use uuid::Uuid;
use sqlx::PgPool;
use serde::{Deserialize, Serialize};
use crate::models::document::Document;
use sqlx::Row;
use reqwest::Client;
use std::env;

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub label: String,
    pub name: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String, // name of source node
    pub target: String, // name of target node
    pub relationship: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Clone)]
pub struct LLMService {
    pool: PgPool,
    client: Client,
    api_key: Option<String>,
    api_url: String,
    model: String,
}

impl LLMService {
    pub fn new(pool: PgPool) -> Self {
        let api_key = env::var("LLM_API_KEY").ok();
        let api_url = env::var("LLM_API_URL").unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
        let model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());

        Self {
            pool,
            client: Client::new(),
            api_key,
            api_url,
            model,
        }
    }

    pub async fn analyze_document(&self, document_id: Uuid) -> Result<GraphData, String> {
        // 1. Fetch document content
        let doc: Document = sqlx::query_as::<_, Document>(
            "SELECT * FROM documents WHERE id = $1"
        )
        .bind(document_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Document not found")?;

        let content = doc.ocr_text.or(doc.content).ok_or("No content to analyze")?;

        // 2. Call LLM
        let graph_data = if self.api_key.is_some() {
            self.call_llm_api(&content).await?
        } else {
            self.mock_llm_response(&content)
        };

        // 3. Store graph data
        self.store_graph_data(document_id, &graph_data).await?;

        Ok(graph_data)
    }

    async fn call_llm_api(&self, content: &str) -> Result<GraphData, String> {
        let prompt = format!(
            "Extract entities (nodes) and relationships (edges) from the following text to build a knowledge graph. \
            Return ONLY a JSON object with two keys: 'nodes' (list of objects with 'label', 'name', 'properties') \
            and 'edges' (list of objects with 'source' (name), 'target' (name), 'relationship', 'properties'). \
            Text: {}",
            content.chars().take(4000).collect::<String>() // Truncate to avoid token limits for now
        );

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts knowledge graphs from text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        });

        let response = self.client.post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key.as_ref().unwrap()))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request to LLM: {}", e))?;

        if !response.status().is_success() {
             return Err(format!("LLM API returned error: {}", response.status()));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        let content_str = response_json["choices"][0]["message"]["content"].as_str()
            .ok_or("Invalid response format from LLM")?;

        // Clean up markdown code blocks if present
        let clean_json = content_str.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let graph_data: GraphData = serde_json::from_str(clean_json)
            .map_err(|e| format!("Failed to parse GraphData JSON: {}", e))?;

        Ok(graph_data)
    }

    fn mock_llm_response(&self, _content: &str) -> GraphData {
        // Mock response
        GraphData {
            nodes: vec![
                GraphNode {
                    label: "Person".to_string(),
                    name: "John Doe".to_string(),
                    properties: serde_json::json!({"role": "Engineer"}),
                },
                GraphNode {
                    label: "Company".to_string(),
                    name: "Readur Corp".to_string(),
                    properties: serde_json::json!({"industry": "Tech"}),
                },
            ],
            edges: vec![
                GraphEdge {
                    source: "John Doe".to_string(),
                    target: "Readur Corp".to_string(),
                    relationship: "WORKS_FOR".to_string(),
                    properties: serde_json::json!({}),
                },
            ],
        }
    }

    async fn store_graph_data(&self, document_id: Uuid, data: &GraphData) -> Result<(), String> {
        let mut tx = self.pool.begin().await.map_err(|e| e.to_string())?;

        // Clear existing graph data for this document
        sqlx::query("DELETE FROM document_nodes WHERE document_id = $1")
            .bind(document_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;

        // Insert nodes
        let mut node_map = std::collections::HashMap::new();

        for node in &data.nodes {
            let row = sqlx::query(
                "INSERT INTO document_nodes (document_id, label, name, properties) VALUES ($1, $2, $3, $4) RETURNING id"
            )
            .bind(document_id)
            .bind(&node.label)
            .bind(&node.name)
            .bind(&node.properties)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;

            let id: Uuid = row.get("id");
            node_map.insert(node.name.clone(), id);
        }

        // Insert edges
        for edge in &data.edges {
            let source_id = node_map.get(&edge.source).ok_or(format!("Source node {} not found", edge.source))?;
            let target_id = node_map.get(&edge.target).ok_or(format!("Target node {} not found", edge.target))?;

            sqlx::query(
                "INSERT INTO document_edges (document_id, source_node_id, target_node_id, relationship, properties) VALUES ($1, $2, $3, $4, $5)"
            )
            .bind(document_id)
            .bind(source_id)
            .bind(target_id)
            .bind(&edge.relationship)
            .bind(&edge.properties)
            .execute(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;
        }

        tx.commit().await.map_err(|e| e.to_string())?;
        Ok(())
    }
}
