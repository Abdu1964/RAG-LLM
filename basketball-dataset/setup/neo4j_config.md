# Setting Up Neo4j for Basketball Dataset

## Prerequisites
- Install Neo4j Desktop or Community Edition.
- Ensure at least 4GB of free RAM for optimal performance.

## Steps
1. Launch Neo4j and create a new database.
2. Open the database and ensure the Cypher query console is accessible.
3. Copy the Cypher script from `data/players_and_teams.cypher`.
4. Paste the script into the Cypher query console and execute.

## Common Issues
- **Error: Out of Memory**: Increase JVM heap size in `neo4j.conf`.
- **Error: Database Lock**: Ensure no other processes are using the database.
