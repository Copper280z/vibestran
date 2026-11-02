#pragma once
#ifdef HAVE_VULKAN
#include "solver/vulkan_buffer.hpp"
#include "core/exceptions.hpp"
#include <format>

namespace vibestran {
// Reused command buffer/fence. begin() records only after the previous
// submission completes; submit()/wait() can enclose independent CPU work.
class VulkanCommands {
public:
    explicit VulkanCommands(const VulkanContext& ctx) : ctx_(ctx) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = ctx.command_pool();
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd_), "allocate command buffer");
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        const auto status = vkCreateFence(ctx.device(), &fi, nullptr, &fence_);
        if (status != VK_SUCCESS) {
            vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmd_);
            check(status, "create fence");
        }
    }
    ~VulkanCommands() {
        drain();
        vkDestroyFence(ctx_.device(), fence_, nullptr);
        vkFreeCommandBuffers(ctx_.device(), ctx_.command_pool(), 1, &cmd_);
    }
    VulkanCommands(const VulkanCommands&) = delete;
    VulkanCommands& operator=(const VulkanCommands&) = delete;

    // Destructors must complete pending work before releasing descriptors or
    // buffers, including when independent host work failed during a submit.
    void drain() noexcept {
        if (pending_) vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
        pending_ = false;
    }

    VkCommandBuffer begin() {
        wait();
        check(vkResetCommandBuffer(cmd_, 0), "reset command buffer");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd_, &bi), "begin command buffer");
        barrier(cmd_);
        return cmd_;
    }
    void submit() {
        check(vkEndCommandBuffer(cmd_), "end command buffer");
        check(vkResetFences(ctx_.device(), 1, &fence_), "reset fence");
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        check(vkQueueSubmit(ctx_.compute_queue(), 1, &si, fence_), "submit");
        pending_ = true;
    }
    void wait() {
        if (!pending_) return;
        check(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX), "wait");
        pending_ = false;
    }
    static void barrier(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    static void copy(VkCommandBuffer cmd, const VulkanBuffer& src,
                     const VulkanBuffer& dst, VkDeviceSize bytes,
                     VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0) {
        const VkBufferCopy region{src_offset, dst_offset, bytes};
        vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
    }
private:
    static void check(VkResult status, const char* action) {
        if (status != VK_SUCCESS)
            throw SolverError(std::format("Vulkan PCG: {} failed (VkResult={})", action, static_cast<int>(status)));
    }
    const VulkanContext& ctx_;
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    VkFence fence_{VK_NULL_HANDLE};
    bool pending_{false};
};
} // namespace vibestran
#endif
