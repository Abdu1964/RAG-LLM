# Basketball Dataset for Neo4j

This project defines a Neo4j graph database representing basketball teams, players, coaches, and game performance statistics. It uses the Cypher query language to create nodes and relationships.

## Folder Structure
- **data/**: Contains Cypher scripts for creating nodes and relationships.
- **setup/**: Guides for configuring Neo4j and running the dataset.

## Usage Instructions
1. Set up Neo4j using the instructions in `setup/neo4j_config.md`.
2. Load the dataset into Neo4j:
   - Open Neo4j Browser.
   - Copy the contents of `data/players_and_teams.cypher`.
   - Paste and run in Neo4j Browser.

3. Query the graph for insights like:
   - Players who scored the most points against a specific team.
   - Coaching relationships for a given team.
   - Teammate relationships within a team.

## Example Queries
- **Find top scorers against a specific team:**
  ```cypher
  MATCH (p:PLAYER)-[r:PLAYED_AGAINST]->(t:TEAM {name: "LA Lakers"})
  RETURN p.name, r.points
  ORDER BY r.points DESC;