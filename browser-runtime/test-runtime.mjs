import { readFile } from "node:fs/promises";
import { policyFromBrowserArtifact } from "./runtime.mjs";

const [artifactPath, probesPath, expectedPath] = process.argv.slice(2);
if (!artifactPath || !probesPath) throw new Error("usage: node test-runtime.mjs ARTIFACT PROBES [EXPECTED_PROVENANCE]");
const expectedProvenance = expectedPath ? JSON.parse(await readFile(expectedPath, "utf8")) : null;
const policy = await policyFromBrowserArtifact(new Uint8Array(await readFile(artifactPath)), expectedProvenance);
const probes = JSON.parse(await readFile(probesPath, "utf8"));
const actual = probes.map(({ history }) => ({ history, output: policy.predict(history) }));
process.stdout.write(JSON.stringify(actual));
