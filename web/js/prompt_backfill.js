import { app } from "../../scripts/app.js";

// After each execution, backfill the final prompt into the last_prompt
// widget so the auto-memory is visible (and survives in saved workflows).
// While lock is on the output equals the base prompt anyway; skipping the
// backfill keeps a manually pinned prompt in the box undisturbed.
app.registerExtension({
    name: "comfyui-ollama-prompt-encode.backfill",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "OllamaPromptGenerator" &&
            nodeData?.name !== "OllamaCLIPTextEncode") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            try {
                const lockWidget = this.widgets?.find((w) => w.name === "lock");
                if (lockWidget?.value) return;
                const text = message?.text?.[0];
                if (typeof text !== "string" || !text) return;
                const widget = this.widgets?.find((w) => w.name === "last_prompt");
                if (widget && widget.value !== text) {
                    widget.value = text;
                }
            } catch (e) {
                console.warn("[comfyui-ollama-prompt-encode] backfill failed", e);
            }
        };
    },
});
