from django.contrib import admin

from .models import PersistentData, SinglePromptChat,MoxieDevice,MoxieSchedule,HiveConfiguration,MentorBehavior,GlobalResponse,ChatTranscript

class ChatTranscriptAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'device', 'module_id', 'content_id', 'role', 'text')
    list_filter = ('device', 'role', 'module_id')
    search_fields = ('text',)

admin.site.register(SinglePromptChat)
admin.site.register(MoxieDevice)
admin.site.register(MoxieSchedule)
admin.site.register(HiveConfiguration)
admin.site.register(MentorBehavior)
admin.site.register(GlobalResponse)
admin.site.register(PersistentData)
admin.site.register(ChatTranscript, ChatTranscriptAdmin)
